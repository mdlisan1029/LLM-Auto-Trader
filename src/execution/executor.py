"""Trade execution via CCXT with paper trading support."""

from __future__ import annotations

import time
from dataclasses import dataclass

import ccxt.async_support as ccxt

from src.execution.order_utils import notional_to_contracts, round_price, round_size
from src.execution.position_manager import PositionManager
from src.llm.response_parser import ActionType, TradeAction
from src.risk.risk_manager import ValidationResult
from src.utils.errors import OrderError
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a trade action."""

    success: bool
    action: str
    symbol: str
    side: str
    size: float = 0
    price: float = 0
    fees: float = 0
    order_id: str = ""
    error: str = ""
    trade_record: dict | None = None


class TradeExecutor:
    """Executes trades on exchange via CCXT.

    Supports both live trading and paper (simulated) trading.
    """

    def __init__(
        self,
        exchange: ccxt.Exchange,
        position_manager: PositionManager,
        paper_mode: bool = True,
        taker_fee: float = 0.0001,
    ):
        self._exchange = exchange
        self._pm = position_manager
        self._paper_mode = paper_mode
        self._taker_fee = taker_fee

    async def execute(
        self,
        action: TradeAction,
        validation: ValidationResult,
        current_price: float,
        equity: float,
        confidence: float | None = None,
        max_holding_s: float | None = None,
    ) -> ExecutionResult:
        """Execute a validated trade action.

        Args:
            action: The LLM's trade action
            validation: Risk manager validation result (must be approved)
            current_price: Current market price for the symbol
            equity: Current account equity
        """
        if not validation.approved:
            return ExecutionResult(
                success=False, action=action.action.value,
                symbol=action.symbol, side="",
                error=f"Rejected: {validation.rejection_reason}",
            )

        try:
            if action.action == ActionType.OPEN_LONG:
                return await self._open_position(action, validation, current_price, equity, "buy", confidence=confidence, max_holding_s=max_holding_s)
            elif action.action == ActionType.OPEN_SHORT:
                return await self._open_position(action, validation, current_price, equity, "sell", confidence=confidence, max_holding_s=max_holding_s)
            elif action.action == ActionType.CLOSE:
                return await self._close_position(action, current_price)
            elif action.action == ActionType.ADJUST_SL:
                return self._adjust_sl(action)
            elif action.action == ActionType.ADJUST_TP:
                return self._adjust_tp(action)
            elif action.action == ActionType.REDUCE_POSITION:
                return await self._reduce_position(action, current_price)
            elif action.action == ActionType.ADD_TO_POSITION:
                return await self._open_position(action, validation, current_price, equity, "buy" if self._get_side(action) == "long" else "sell", confidence=confidence, max_holding_s=max_holding_s)
            elif action.action == ActionType.HOLD:
                return ExecutionResult(success=True, action="hold", symbol=action.symbol, side="")
            else:
                return ExecutionResult(
                    success=False, action=action.action.value,
                    symbol=action.symbol, side="",
                    error=f"Unknown action: {action.action}",
                )
        except Exception as e:
            log.error("execution_error", action=action.action.value, symbol=action.symbol, error=str(e))
            return ExecutionResult(
                success=False, action=action.action.value,
                symbol=action.symbol, side="", error=str(e),
            )

    async def _open_position(
        self,
        action: TradeAction,
        validation: ValidationResult,
        current_price: float,
        equity: float,
        ccxt_side: str,
        confidence: float | None = None,
        max_holding_s: float | None = None,
    ) -> ExecutionResult:
        """Open a new position."""
        size_pct = validation.adjusted_size_pct or action.size_pct or 10
        leverage = action.leverage or 2
        notional = equity * (size_pct / 100)

        # Get market info for rounding
        market = self._exchange.market(action.symbol)
        lot_size = market.get("precision", {}).get("amount", 0.001)
        tick_size = market.get("precision", {}).get("price", 0.01)

        # Convert notional to contracts
        contracts = notional_to_contracts(notional, current_price)
        contracts = round_size(contracts, lot_size)

        if contracts <= 0:
            return ExecutionResult(
                success=False, action=action.action.value,
                symbol=action.symbol, side=ccxt_side,
                error="Position size too small after rounding",
            )

        side = "long" if ccxt_side == "buy" else "short"
        entry_price = current_price

        if self._paper_mode:
            # Paper trade: simulate fill at current price
            fees = notional * self._taker_fee
            self._pm.open_position(
                symbol=action.symbol, side=side, size=contracts,
                notional=notional, entry_price=entry_price,
                leverage=leverage, stop_loss=action.stop_loss,
                take_profit=action.take_profit, fees=fees,
                reasoning=action.reasoning,
                llm_confidence=confidence,
                max_holding_s=max_holding_s,
            )
            return ExecutionResult(
                success=True, action=action.action.value,
                symbol=action.symbol, side=side,
                size=contracts, price=entry_price,
                fees=fees, order_id=f"paper_{int(time.time())}",
            )
        else:
            # Live trade
            try:
                # Set leverage first. Some CCXT exchanges (or markets within
                # an exchange) do not expose set_leverage; tolerate that and
                # log instead of crashing — the order will still be placed at
                # the exchange's account-default leverage.
                try:
                    await self._exchange.set_leverage(leverage, action.symbol)
                except Exception as lev_err:  # ccxt.NotSupported, AttributeError, ...
                    log.warning(
                        "set_leverage_unsupported",
                        symbol=action.symbol,
                        leverage=leverage,
                        error=str(lev_err),
                    )

                # Place order
                if action.order_type == "limit" and action.entry_price:
                    price = round_price(action.entry_price, tick_size)
                    order = await self._exchange.create_order(
                        action.symbol, "limit", ccxt_side, contracts, price,
                    )
                else:
                    order = await self._exchange.create_order(
                        action.symbol, "market", ccxt_side, contracts,
                    )

                fill_price = order.get("average", current_price) or current_price
                fill_size = order.get("filled", contracts) or contracts
                fees = order.get("fee", {}).get("cost", notional * self._taker_fee) or 0

                self._pm.open_position(
                    symbol=action.symbol, side=side, size=fill_size,
                    notional=fill_size * fill_price, entry_price=fill_price,
                    leverage=leverage, stop_loss=action.stop_loss,
                    take_profit=action.take_profit, fees=fees,
                    reasoning=action.reasoning,
                    llm_confidence=confidence,
                    max_holding_s=max_holding_s,
                )

                return ExecutionResult(
                    success=True, action=action.action.value,
                    symbol=action.symbol, side=side,
                    size=fill_size, price=fill_price,
                    fees=fees, order_id=order.get("id", ""),
                )
            except ccxt.InsufficientFunds as e:
                raise OrderError(f"Insufficient funds: {e}") from e
            except ccxt.ExchangeError as e:
                raise OrderError(f"Exchange error: {e}") from e

    async def close_position_for_trigger(
        self,
        symbol: str,
        current_price: float,
        reason: str,
    ) -> ExecutionResult:
        """Close an SL/TP-triggered position through the exchange executor.

        Unlike the old local-only trigger handling, this method submits a
        reduce-only market order in live mode. The position manager is updated
        by _close_position only after the exchange order call succeeds.
        """
        action = TradeAction(
            action=ActionType.CLOSE,
            symbol=symbol,
            reasoning=f"Automatic {reason.replace('_', ' ')} close",
        )
        try:
            result = await self._close_position(action, current_price)
            if not result.success:
                log.error(
                    "triggered_position_close_failed",
                    symbol=symbol,
                    reason=reason,
                    error=result.error,
                )
            return result
        except Exception as e:
            log.error(
                "triggered_position_close_failed",
                symbol=symbol,
                reason=reason,
                error=str(e),
                exc_info=True,
            )
            return ExecutionResult(
                success=False,
                action="close",
                symbol=symbol,
                side="",
                error=str(e),
            )

    async def _close_position(
        self, action: TradeAction, current_price: float
    ) -> ExecutionResult:
        """Close an existing position."""
        pos = self._pm.get_position(action.symbol)
        if pos is None:
            return ExecutionResult(
                success=False, action="close", symbol=action.symbol, side="",
                error=f"No position found for {action.symbol}",
            )

        ccxt_side = "sell" if pos.side == "long" else "buy"

        if self._paper_mode:
            fees = pos.notional * self._taker_fee
            trade_record = self._pm.close_position(action.symbol, current_price, fees)
            return ExecutionResult(
                success=True, action="close", symbol=action.symbol,
                side=pos.side, size=pos.size, price=current_price,
                fees=fees, order_id=f"paper_{int(time.time())}",
                trade_record=trade_record or None,
            )
        else:
            try:
                order = await self._exchange.create_order(
                    action.symbol, "market", ccxt_side, pos.size,
                    params={"reduceOnly": True},
                )
                fill_price = order.get("average", current_price) or current_price
                fees = order.get("fee", {}).get("cost", 0) or 0
                trade_record = self._pm.close_position(action.symbol, fill_price, fees)
                return ExecutionResult(
                    success=True, action="close", symbol=action.symbol,
                    side=pos.side, size=pos.size, price=fill_price,
                    fees=fees, order_id=order.get("id", ""),
                    trade_record=trade_record or None,
                )
            except ccxt.ExchangeError as e:
                raise OrderError(f"Close failed: {e}") from e

    async def _reduce_position(
        self, action: TradeAction, current_price: float
    ) -> ExecutionResult:
        """Reduce position size (partial close)."""
        pos = self._pm.get_position(action.symbol)
        if pos is None:
            return ExecutionResult(
                success=False, action="reduce", symbol=action.symbol, side="",
                error=f"No position for {action.symbol}",
            )

        reduce_pct = (action.size_pct or 50) / 100
        reduce_size = pos.size * reduce_pct
        ccxt_side = "sell" if pos.side == "long" else "buy"

        if self._paper_mode:
            fees = reduce_size * current_price * self._taker_fee
            # Update position size
            pos.size -= reduce_size
            pos.notional -= reduce_size * current_price
            pos.fees_paid += fees
            if pos.size <= 0:
                self._pm.close_position(action.symbol, current_price, fees)
            return ExecutionResult(
                success=True, action="reduce", symbol=action.symbol,
                side=pos.side, size=reduce_size, price=current_price,
                fees=fees,
            )
        else:
            market = self._exchange.market(action.symbol)
            lot_size = market.get("precision", {}).get("amount", 0.001)
            reduce_size = round_size(reduce_size, lot_size)
            order = await self._exchange.create_order(
                action.symbol, "market", ccxt_side, reduce_size,
                params={"reduceOnly": True},
            )
            fill_price = order.get("average", current_price) or current_price
            fees = order.get("fee", {}).get("cost", 0) or 0
            pos.size -= reduce_size
            pos.notional -= reduce_size * fill_price
            pos.fees_paid += fees
            if pos.size <= 0:
                self._pm.close_position(action.symbol, fill_price, fees)
            return ExecutionResult(
                success=True, action="reduce", symbol=action.symbol,
                side=pos.side, size=reduce_size, price=fill_price,
                fees=fees, order_id=order.get("id", ""),
            )

    def _adjust_sl(self, action: TradeAction) -> ExecutionResult:
        """Adjust stop-loss for a position."""
        if action.stop_loss is not None:
            self._pm.update_stop_loss(action.symbol, action.stop_loss)
            return ExecutionResult(
                success=True, action="adjust_sl", symbol=action.symbol, side="",
                price=action.stop_loss,
            )
        return ExecutionResult(
            success=False, action="adjust_sl", symbol=action.symbol, side="",
            error="No stop_loss value provided",
        )

    def _adjust_tp(self, action: TradeAction) -> ExecutionResult:
        """Adjust take-profit for a position."""
        if action.take_profit is not None:
            self._pm.update_take_profit(action.symbol, action.take_profit)
            return ExecutionResult(
                success=True, action="adjust_tp", symbol=action.symbol, side="",
                price=action.take_profit,
            )
        return ExecutionResult(
            success=False, action="adjust_tp", symbol=action.symbol, side="",
            error="No take_profit value provided",
        )

    def _get_side(self, action: TradeAction) -> str:
        """Determine side from action type."""
        if action.action in (ActionType.OPEN_LONG, ActionType.ADD_TO_POSITION):
            return "long"
        return "short"

    async def emergency_close_all(self) -> list[ExecutionResult]:
        """Emergency: close all positions immediately. Used by circuit breakers."""
        results = []
        for pos in self._pm.open_positions:
            try:
                if self._paper_mode:
                    self._pm.close_position(pos.symbol, pos.mark_price, 0)
                    results.append(ExecutionResult(
                        success=True, action="emergency_close",
                        symbol=pos.symbol, side=pos.side,
                        size=pos.size, price=pos.mark_price,
                    ))
                else:
                    ccxt_side = "sell" if pos.side == "long" else "buy"
                    order = await self._exchange.create_order(
                        pos.symbol, "market", ccxt_side, pos.size,
                        params={"reduceOnly": True},
                    )
                    fill_price = order.get("average", pos.mark_price) or pos.mark_price
                    fees = order.get("fee", {}).get("cost", 0) or 0
                    self._pm.close_position(pos.symbol, fill_price, fees)
                    results.append(ExecutionResult(
                        success=True, action="emergency_close",
                        symbol=pos.symbol, side=pos.side,
                        size=pos.size, price=fill_price, fees=fees,
                        order_id=order.get("id", ""),
                    ))
                log.critical("emergency_close", symbol=pos.symbol, side=pos.side)
            except Exception as e:
                log.error("emergency_close_failed", symbol=pos.symbol, error=str(e))
                results.append(ExecutionResult(
                    success=False, action="emergency_close",
                    symbol=pos.symbol, side=pos.side, error=str(e),
                ))
        return results
