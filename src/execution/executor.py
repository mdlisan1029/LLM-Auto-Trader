"""Trade execution via CCXT with paper trading support."""

from __future__ import annotations

import math
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
    order_status: str = ""
    filled_size: float = 0
    remaining_size: float = 0
    # True when the exchange may have accepted the order but its final state
    # could not be confirmed. Remaining size is then a conservative upper bound.
    state_uncertain: bool = False


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

    @staticmethod
    def _optional_nonnegative_float(value: object) -> float | None:
        """Parse a finite, non-negative numeric order field."""
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 0:
            return None
        return number

    async def _resolve_order_fill(
        self,
        order: dict,
        symbol: str,
        requested_size: float,
    ) -> tuple[dict, float, bool]:
        """Resolve a terminal exchange order and its actual filled quantity.

        Open orders are canceled before local state is changed. When available,
        fetch_order is used after cancellation to capture any fill that raced
        with the cancel request. Missing or ambiguous fill data is rejected.
        """
        if not isinstance(order, dict):
            raise OrderError("Exchange returned an invalid order response")
        if requested_size <= 0 or not math.isfinite(requested_size):
            raise OrderError(f"Invalid requested order size: {requested_size}")

        resolved = dict(order)
        order_id = resolved.get("id")
        status = str(resolved.get("status") or "").lower()
        terminal_statuses = {"closed", "canceled", "cancelled", "expired", "rejected"}

        if status in {"open", "pending", "new"}:
            if not order_id:
                raise OrderError(
                    f"Order for {symbol} is {status} but has no id; fill cannot be reconciled"
                )

            cancel_order = getattr(self._exchange, "cancel_order", None)
            if callable(cancel_order):
                try:
                    canceled = await cancel_order(order_id, symbol)
                    if isinstance(canceled, dict):
                        resolved.update(canceled)
                except Exception as cancel_error:
                    # The order may have filled just before cancellation.
                    log.warning(
                        "order_cancel_for_fill_reconciliation_failed",
                        symbol=symbol,
                        order_id=order_id,
                        error=str(cancel_error),
                    )

            fetch_order = getattr(self._exchange, "fetch_order", None)
            if callable(fetch_order):
                try:
                    latest = await fetch_order(order_id, symbol)
                    if isinstance(latest, dict):
                        resolved.update(latest)
                except Exception as fetch_error:
                    log.warning(
                        "order_fetch_after_cancel_failed",
                        symbol=symbol,
                        order_id=order_id,
                        error=str(fetch_error),
                    )

        elif status not in terminal_statuses and order_id:
            # Some exchange adapters return incomplete create_order payloads.
            fetch_order = getattr(self._exchange, "fetch_order", None)
            if callable(fetch_order):
                try:
                    latest = await fetch_order(order_id, symbol)
                    if isinstance(latest, dict):
                        resolved.update(latest)
                except Exception as fetch_error:
                    log.warning(
                        "order_fetch_for_fill_verification_failed",
                        symbol=symbol,
                        order_id=order_id,
                        error=str(fetch_error),
                    )

        status = str(resolved.get("status") or "").lower()
        if status not in terminal_statuses:
            raise OrderError(
                f"Order {order_id or '<unknown>'} for {symbol} is not in a confirmed "
                f"terminal state (status={status or 'missing'}); local position was not changed"
            )

        amount = self._optional_nonnegative_float(resolved.get("amount"))
        filled = self._optional_nonnegative_float(resolved.get("filled"))
        if filled is None:
            # CCXT defines "closed" as fully filled. Only use the order amount
            # when the exchange explicitly reports that terminal status.
            if status == "closed" and amount is not None:
                filled = amount
            else:
                raise OrderError(
                    f"Order {order_id or '<unknown>'} for {symbol} has no reliable "
                    "filled quantity; local position was not changed"
                )

        tolerance = max(requested_size * 1e-8, 1e-12)
        if filled > requested_size + tolerance:
            raise OrderError(
                f"Exchange reports filled={filled} above requested={requested_size} "
                f"for {symbol}; local position was not changed"
            )

        filled = min(filled, requested_size)
        fully_filled = filled > 0 and filled + tolerance >= requested_size
        return resolved, filled, fully_filled

    def _order_fill_price(self, order: dict, fallback_price: float) -> float:
        average = self._optional_nonnegative_float(order.get("average"))
        if average is not None and average > 0:
            return average
        if fallback_price > 0 and math.isfinite(fallback_price):
            return fallback_price
        raise OrderError("Exchange fill price is unavailable and no valid market price exists")

    def _order_fee(self, order: dict, fallback_fee: float) -> float:
        fee = order.get("fee") or {}
        fee_value = self._optional_nonnegative_float(
            fee.get("cost") if isinstance(fee, dict) else None
        )
        return fee_value if fee_value is not None else max(0.0, fallback_fee)

    def _unconfirmed_order_result(
        self,
        *,
        action: str,
        symbol: str,
        side: str,
        requested_size: float,
        error: Exception,
        order: dict | None = None,
    ) -> ExecutionResult:
        """Keep uncertain exchange outcomes visible and block unsafe assumptions."""
        order = order if isinstance(order, dict) else {}
        order_id = str(order.get("id") or "")
        status = str(order.get("status") or "unknown").lower()
        reported_fill = self._optional_nonnegative_float(order.get("filled"))
        fill_note = (
            f" Exchange's non-terminal response reported filled={reported_fill},"
            " which is not treated as a confirmed fill."
            if reported_fill is not None else ""
        )
        return ExecutionResult(
            success=False,
            action=action,
            symbol=symbol,
            side=side,
            size=requested_size,
            order_id=order_id,
            order_status=status,
            filled_size=0.0,
            # This is the maximum quantity that could still be exposed, not
            # a claim that the full amount definitely remains open.
            remaining_size=requested_size,
            state_uncertain=True,
            error=(
                f"Exchange order outcome is unknown (status={status}). No fill "
                f"was applied locally; up to {requested_size} may remain exposed. "
                f"Reconcile the exchange order/position before trading again. "
                f"Cause: {error}.{fill_note}"
            ),
        )

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

                # Place and resolve the order. A timeout or an ambiguous
                # response can happen after the exchange accepted the request.
                order: dict | None = None
                try:
                    if action.order_type == "limit" and action.entry_price:
                        price = round_price(action.entry_price, tick_size)
                        order = await self._exchange.create_order(
                            action.symbol, "limit", ccxt_side, contracts, price,
                        )
                    else:
                        order = await self._exchange.create_order(
                            action.symbol, "market", ccxt_side, contracts,
                        )

                    order, fill_size, fully_filled = await self._resolve_order_fill(
                        order, action.symbol, contracts,
                    )
                except Exception as order_error:
                    return self._unconfirmed_order_result(
                        action=action.action.value,
                        symbol=action.symbol,
                        side=side,
                        requested_size=contracts,
                        error=order_error,
                        order=order,
                    )
                order_id = str(order.get("id") or "")
                order_status = str(order.get("status") or "").lower()
                if fill_size <= 0:
                    return ExecutionResult(
                        success=False, action=action.action.value,
                        symbol=action.symbol, side=side,
                        order_id=order_id, order_status=order_status,
                        filled_size=0, remaining_size=contracts,
                        error=f"Entry order was {order_status} with zero confirmed fills",
                    )

                fill_price = self._order_fill_price(order, current_price)
                fees = self._order_fee(
                    order, fill_size * fill_price * self._taker_fee,
                )

                self._pm.open_position(
                    symbol=action.symbol, side=side, size=fill_size,
                    notional=fill_size * fill_price, entry_price=fill_price,
                    leverage=leverage, stop_loss=action.stop_loss,
                    take_profit=action.take_profit, fees=fees,
                    reasoning=action.reasoning,
                    llm_confidence=confidence,
                    max_holding_s=max_holding_s,
                )

                remaining_size = max(0.0, contracts - fill_size)
                if not fully_filled:
                    log.warning(
                        "entry_order_partially_filled",
                        symbol=action.symbol,
                        order_id=order_id,
                        requested_size=contracts,
                        filled_size=fill_size,
                        remaining_size=remaining_size,
                    )
                return ExecutionResult(
                    success=True, action=action.action.value,
                    symbol=action.symbol, side=side,
                    size=fill_size, price=fill_price,
                    fees=fees, order_id=order_id,
                    order_status=order_status,
                    filled_size=fill_size, remaining_size=remaining_size,
                    error="" if fully_filled else (
                        f"Partial entry fill: {fill_size} of {contracts}; "
                        "unfilled remainder was canceled"
                    ),
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
            order: dict | None = None
            try:
                order = await self._exchange.create_order(
                    action.symbol, "market", ccxt_side, pos.size,
                    params={"reduceOnly": True},
                )
                order, filled_size, fully_filled = await self._resolve_order_fill(
                    order, action.symbol, pos.size,
                )
                order_id = str(order.get("id") or "")
                order_status = str(order.get("status") or "").lower()
                if filled_size <= 0:
                    return ExecutionResult(
                        success=False, action="close", symbol=action.symbol,
                        side=pos.side, order_id=order_id, order_status=order_status,
                        filled_size=0, remaining_size=pos.size,
                        error=f"Close order was {order_status} with zero confirmed fills",
                    )

                fill_price = self._order_fill_price(order, current_price)
                fees = self._order_fee(
                    order, filled_size * fill_price * self._taker_fee,
                )
                if fully_filled:
                    trade_record = self._pm.close_position(
                        action.symbol, fill_price, fees,
                    )
                    return ExecutionResult(
                        success=True, action="close", symbol=action.symbol,
                        side=pos.side, size=filled_size, price=fill_price,
                        fees=fees, order_id=order_id,
                        order_status=order_status, filled_size=filled_size,
                        remaining_size=0, trade_record=trade_record or None,
                    )

                remaining_size = self._pm.reduce_position(
                    action.symbol, filled_size, fill_price, fees,
                )
                log.warning(
                    "close_order_partially_filled",
                    symbol=action.symbol,
                    order_id=order_id,
                    requested_size=pos.size,
                    filled_size=filled_size,
                    remaining_size=remaining_size,
                )
                return ExecutionResult(
                    success=False, action="close", symbol=action.symbol,
                    side=pos.side, size=filled_size, price=fill_price,
                    fees=fees, order_id=order_id,
                    order_status=order_status, filled_size=filled_size,
                    remaining_size=remaining_size,
                    error=(
                        f"Partial close fill: {filled_size} of {pos.size}; "
                        f"{remaining_size} remains open"
                    ),
                )
            except Exception as e:
                return self._unconfirmed_order_result(
                    action="close",
                    symbol=action.symbol,
                    side=pos.side,
                    requested_size=pos.size,
                    error=e,
                    order=order,
                )

    async def _reduce_position(
        self, action: TradeAction, current_price: float
    ) -> ExecutionResult:
        """Reduce position size using only exchange-confirmed filled quantity."""
        pos = self._pm.get_position(action.symbol)
        if pos is None:
            return ExecutionResult(
                success=False, action="reduce", symbol=action.symbol, side="",
                error=f"No position for {action.symbol}",
            )

        reduce_pct = (action.size_pct or 50) / 100
        requested_size = min(pos.size, pos.size * reduce_pct)
        ccxt_side = "sell" if pos.side == "long" else "buy"

        if self._paper_mode:
            filled_size = requested_size
            fill_price = current_price
            fees = filled_size * fill_price * self._taker_fee
            order_id = f"paper_{int(time.time())}"
            order_status = "closed"
            fully_filled = True
        else:
            market = self._exchange.market(action.symbol)
            lot_size = market.get("precision", {}).get("amount", 0.001)
            requested_size = round_size(requested_size, lot_size)
            if requested_size <= 0:
                return ExecutionResult(
                    success=False, action="reduce", symbol=action.symbol,
                    side=pos.side, error="Reduce size is too small after rounding",
                )
            order: dict | None = None
            try:
                order = await self._exchange.create_order(
                    action.symbol, "market", ccxt_side, requested_size,
                    params={"reduceOnly": True},
                )
                order, filled_size, fully_filled = await self._resolve_order_fill(
                    order, action.symbol, requested_size,
                )
            except Exception as order_error:
                return self._unconfirmed_order_result(
                    action="reduce",
                    symbol=action.symbol,
                    side=pos.side,
                    requested_size=requested_size,
                    error=order_error,
                    order=order,
                )
            order_id = str(order.get("id") or "")
            order_status = str(order.get("status") or "").lower()
            if filled_size <= 0:
                return ExecutionResult(
                    success=False, action="reduce", symbol=action.symbol,
                    order_id=order_id, order_status=order_status,
                    filled_size=0, remaining_size=pos.size,
                    error=f"Reduce order was {order_status} with zero confirmed fills",
                )
            fill_price = self._order_fill_price(order, current_price)
            fees = self._order_fee(
                order, filled_size * fill_price * self._taker_fee,
            )

        tolerance = max(pos.size * 1e-8, 1e-12)
        trade_record = None
        if filled_size + tolerance >= pos.size:
            trade_record = self._pm.close_position(action.symbol, fill_price, fees)
            remaining_size = 0.0
        else:
            remaining_size = self._pm.reduce_position(
                action.symbol, filled_size, fill_price, fees,
            )

        if not fully_filled:
            log.warning(
                "reduce_order_partially_filled",
                symbol=action.symbol,
                order_id=order_id,
                requested_size=requested_size,
                filled_size=filled_size,
                remaining_position_size=remaining_size,
            )

        return ExecutionResult(
            success=fully_filled, action="reduce", symbol=action.symbol,
            side=pos.side, size=filled_size, price=fill_price,
            fees=fees, order_id=order_id, order_status=order_status,
            filled_size=filled_size, remaining_size=remaining_size,
            trade_record=trade_record or None,
            error="" if fully_filled else (
                f"Partial reduce fill: {filled_size} of {requested_size}; "
                "position was updated only for the confirmed fill"
            ),
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
        """Emergency close: update local positions only for confirmed fills."""
        results: list[ExecutionResult] = []
        for pos in self._pm.open_positions:
            order: dict | None = None
            try:
                if self._paper_mode:
                    trade_record = self._pm.close_position(pos.symbol, pos.mark_price, 0)
                    results.append(ExecutionResult(
                        success=True, action="emergency_close",
                        symbol=pos.symbol, side=pos.side,
                        size=pos.size, price=pos.mark_price,
                        filled_size=pos.size, remaining_size=0,
                        trade_record=trade_record or None,
                        order_status="closed",
                    ))
                else:
                    ccxt_side = "sell" if pos.side == "long" else "buy"
                    order = await self._exchange.create_order(
                        pos.symbol, "market", ccxt_side, pos.size,
                        params={"reduceOnly": True},
                    )
                    order, filled_size, fully_filled = await self._resolve_order_fill(
                        order, pos.symbol, pos.size,
                    )
                    order_id = str(order.get("id") or "")
                    order_status = str(order.get("status") or "").lower()
                    if filled_size <= 0:
                        results.append(ExecutionResult(
                            success=False, action="emergency_close",
                            symbol=pos.symbol, side=pos.side,
                            order_id=order_id, order_status=order_status,
                            filled_size=0, remaining_size=pos.size,
                            error=f"Emergency close was {order_status} with zero confirmed fills",
                        ))
                        continue

                    fill_price = self._order_fill_price(order, pos.mark_price)
                    fees = self._order_fee(
                        order, filled_size * fill_price * self._taker_fee,
                    )
                    if fully_filled:
                        trade_record = self._pm.close_position(pos.symbol, fill_price, fees)
                        remaining_size = 0.0
                    else:
                        trade_record = None
                        remaining_size = self._pm.reduce_position(
                            pos.symbol, filled_size, fill_price, fees,
                        )

                    results.append(ExecutionResult(
                        success=fully_filled, action="emergency_close",
                        symbol=pos.symbol, side=pos.side,
                        size=filled_size, price=fill_price, fees=fees,
                        order_id=order_id, order_status=order_status,
                        filled_size=filled_size, remaining_size=remaining_size,
                        trade_record=trade_record or None,
                        error="" if fully_filled else (
                            f"Emergency close partially filled: {filled_size} of {pos.size}; "
                            f"{remaining_size} remains open"
                        ),
                    ))
                log.critical("emergency_close", symbol=pos.symbol, side=pos.side)
            except Exception as e:
                log.error("emergency_close_failed", symbol=pos.symbol, error=str(e))
                results.append(self._unconfirmed_order_result(
                    action="emergency_close",
                    symbol=pos.symbol,
                    side=pos.side,
                    requested_size=pos.size,
                    error=e,
                    order=order,
                ))
        return results
