"""Position tracking and portfolio state management."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Position:
    """Tracked position with full state."""

    symbol: str
    side: str  # "long" or "short"
    size: float  # in contracts
    notional: float  # in USDT
    entry_price: float
    leverage: int
    stop_loss: float | None = None
    take_profit: float | None = None

    # Dynamic state
    mark_price: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    fees_paid: float = 0

    # Tracking
    highest_price: float = 0  # for trailing SL
    lowest_price: float = 0
    opened_at: float = field(default_factory=time.time)
    llm_reasoning: str = ""

    # Partial close tracking
    original_size: float = 0
    partial_tp_taken: bool = False

    # Phase 1 §3.5 — triple-barrier config (set at open; consumed at close)
    pt_pct: float | None = None
    sl_pct: float | None = None
    max_holding_s: float | None = None
    llm_confidence: float | None = None

    # Phase 3 §3B.7 — cumulative funding bleed tracker
    funding_paid_usdt: float = 0.0
    last_funding_update_ts: float = 0.0

    def accrue_funding(self, funding_rate_per_8h: float, now: float) -> float:
        """Accrue fractional funding paid since last update. Returns USDT paid
        this call (may be zero). Negative rate on a long = receives (we record
        as negative USDT paid).
        """
        if self.last_funding_update_ts == 0:
            self.last_funding_update_ts = now
            return 0.0
        elapsed = max(0.0, now - self.last_funding_update_ts)
        if elapsed <= 0:
            return 0.0
        # For long: pays when rate > 0. For short: pays when rate < 0.
        sign = 1.0 if self.side == "long" else -1.0
        per_second_rate = funding_rate_per_8h / (8.0 * 3600.0)
        paid = sign * per_second_rate * self.notional * elapsed
        self.funding_paid_usdt += paid
        self.last_funding_update_ts = now
        return paid

    def update_price(self, price: float) -> None:
        """Update position with latest market price."""
        self.mark_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price) if self.lowest_price > 0 else price

        # Calculate unrealized PnL
        if self.side == "long":
            self.unrealized_pnl = (price - self.entry_price) / self.entry_price * self.notional
        else:
            self.unrealized_pnl = (self.entry_price - price) / self.entry_price * self.notional

    @property
    def pnl_pct(self) -> float:
        """Unrealized PnL as % of position margin."""
        margin = self.notional / self.leverage if self.leverage else self.notional
        return (self.unrealized_pnl / margin * 100) if margin else 0

    @property
    def duration_seconds(self) -> float:
        return time.time() - self.opened_at

    @property
    def duration_str(self) -> str:
        d = self.duration_seconds
        if d < 60:
            return f"{d:.0f}s"
        if d < 3600:
            return f"{d/60:.0f}m"
        return f"{d/3600:.1f}h"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "notional": self.notional,
            "entry_price": self.entry_price,
            "mark_price": self.mark_price,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "pnl_pct": round(self.pnl_pct, 2),
            "fees_paid": self.fees_paid,
            "duration": self.duration_str,
            "opened_at": self.opened_at,
        }


class PositionManager:
    """Manages all open positions and portfolio-level state."""

    def __init__(self):
        self._positions: dict[str, Position] = {}  # symbol -> Position
        self._peak_equity: float = 0
        self._starting_equity: float = 0

    def set_starting_equity(self, equity: float) -> None:
        self._starting_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

    def sync_from_exchange(self, exchange_positions: list[dict]) -> None:
        """Sync positions from exchange data (startup recovery).

        Only adds positions not already tracked. Does not overwrite
        locally tracked positions with exchange data.
        """
        for ep in exchange_positions:
            sym = ep["symbol"]
            if sym not in self._positions:
                pos = Position(
                    symbol=sym,
                    side=ep.get("side", "long"),
                    size=ep.get("size", 0),
                    notional=ep.get("notional", 0),
                    entry_price=ep.get("entry_price", 0),
                    leverage=ep.get("leverage", 1),
                    mark_price=ep.get("mark_price", 0),
                    unrealized_pnl=ep.get("unrealized_pnl", 0),
                )
                pos.original_size = pos.size
                pos.highest_price = pos.mark_price
                pos.lowest_price = pos.mark_price
                self._positions[sym] = pos
                log.info("position_synced", symbol=sym, side=pos.side, notional=pos.notional)

    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        notional: float,
        entry_price: float,
        leverage: int,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        fees: float = 0,
        reasoning: str = "",
        max_holding_s: float | None = None,
        llm_confidence: float | None = None,
    ) -> Position:
        """Record a new position opening."""
        pt_pct = None
        sl_pct = None
        if entry_price and entry_price > 0:
            if take_profit is not None:
                pt_pct = abs(take_profit - entry_price) / entry_price
            if stop_loss is not None:
                sl_pct = abs(stop_loss - entry_price) / entry_price

        pos = Position(
            symbol=symbol,
            side=side,
            size=size,
            notional=notional,
            entry_price=entry_price,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            fees_paid=fees,
            mark_price=entry_price,
            highest_price=entry_price,
            lowest_price=entry_price,
            original_size=size,
            llm_reasoning=reasoning,
            pt_pct=pt_pct,
            sl_pct=sl_pct,
            max_holding_s=max_holding_s,
            llm_confidence=llm_confidence,
        )
        self._positions[symbol] = pos
        log.info("position_opened", symbol=symbol, side=side, notional=notional,
                 leverage=leverage, entry=entry_price, sl=stop_loss, tp=take_profit,
                 llm_confidence=llm_confidence, pt_pct=pt_pct, sl_pct=sl_pct)
        return pos

    def close_position(self, symbol: str, exit_price: float, fees: float = 0) -> dict:
        """Close a position and return trade record."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            log.warning("close_position_not_found", symbol=symbol)
            return {}

        # Include PnL realized by earlier partial fills.
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) / pos.entry_price * pos.notional
        else:
            pnl = (pos.entry_price - exit_price) / pos.entry_price * pos.notional
        pnl += pos.realized_pnl

        total_fees = pos.fees_paid + fees
        net_pnl = pnl - total_fees
        pnl_pct = (net_pnl / (pos.notional / pos.leverage)) * 100 if pos.notional else 0

        trade = {
            "symbol": symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "size": pos.size,
            "notional": pos.notional,
            "leverage": pos.leverage,
            "pnl": round(net_pnl, 4),
            "pnl_pct": round(pnl_pct, 2),
            "fees": round(total_fees, 4),
            "duration_seconds": pos.duration_seconds,
            "timestamp_open": pos.opened_at,
            "timestamp_close": time.time(),
            "reasoning": pos.llm_reasoning,
            "llm_confidence": pos.llm_confidence,
            "pt_pct": pos.pt_pct,
            "sl_pct": pos.sl_pct,
            "max_holding_s": pos.max_holding_s,
        }

        # Phase 1 §3.5 — triple-barrier labeling. We don't have a full price path
        # here (see design doc §3.5); use a close-time inference based on the
        # highest/lowest prices reached and the exit vs barriers.
        label_info = _infer_triple_barrier_label(pos, exit_price)
        if label_info is not None:
            trade["tb_label"] = label_info["label"]
            trade["tb_barrier_touched"] = label_info["barrier_touched"]
            trade["tb_return_at_touch"] = label_info["return_at_touch"]

        log.info(
            "position_closed",
            **{k: v for k, v in trade.items() if k != "reasoning"},
        )
        return trade

    def reduce_position(
        self,
        symbol: str,
        reduced_size: float,
        exit_price: float,
        fees: float = 0,
    ) -> float:
        """Apply an exchange-confirmed partial close and return remaining size.

        Realized PnL and fees are retained on the open position so the final
        close record includes every confirmed partial execution.
        """
        pos = self._positions.get(symbol)
        if pos is None or reduced_size <= 0 or pos.size <= 0:
            return pos.size if pos is not None else 0.0

        actual_size = min(float(reduced_size), pos.size)
        fraction = actual_size / pos.size
        closed_notional = pos.notional * fraction

        if pos.entry_price > 0:
            if pos.side == "long":
                realized = (exit_price - pos.entry_price) / pos.entry_price * closed_notional
            else:
                realized = (pos.entry_price - exit_price) / pos.entry_price * closed_notional
            pos.realized_pnl += realized

        pos.size = max(0.0, pos.size - actual_size)
        pos.notional = max(0.0, pos.notional - closed_notional)
        pos.fees_paid += max(0.0, fees)

        # Avoid leaving a floating-point dust position after a full fill.
        tolerance = max(actual_size * 1e-10, 1e-12)
        if pos.size <= tolerance:
            pos.size = 0.0
            pos.notional = 0.0

        log.info(
            "position_partially_reduced",
            symbol=symbol,
            reduced_size=actual_size,
            remaining_size=pos.size,
            realized_pnl=pos.realized_pnl,
            fees_paid=pos.fees_paid,
        )
        return pos.size

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update all position prices from latest market data."""
        for sym, pos in self._positions.items():
            if sym in prices:
                pos.update_price(prices[sym])

    def update_stop_loss(self, symbol: str, new_sl: float) -> None:
        """Update stop-loss for a position."""
        if symbol in self._positions:
            self._positions[symbol].stop_loss = new_sl

    def update_take_profit(self, symbol: str, new_tp: float) -> None:
        """Update take-profit for a position."""
        if symbol in self._positions:
            self._positions[symbol].take_profit = new_tp

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    @property
    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    @property
    def position_count(self) -> int:
        return len(self._positions)

    @property
    def has_positions(self) -> bool:
        return bool(self._positions)

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    @property
    def total_exposure(self) -> float:
        """Total notional exposure across all positions."""
        return sum(abs(p.notional) for p in self._positions.values())

    @property
    def total_exposure_pct(self) -> float:
        """Total exposure as % of starting equity."""
        if self._starting_equity <= 0:
            return 0
        return self.total_exposure / self._starting_equity

    def get_current_prices(self) -> dict[str, float]:
        """Get current mark prices for all positions."""
        return {sym: pos.mark_price for sym, pos in self._positions.items()}

    def positions_for_prompt(self) -> list[dict]:
        """Get position data formatted for LLM prompt."""
        return [p.to_dict() for p in self._positions.values()]

    def check_stop_losses(self, prices: dict[str, float]) -> list[str]:
        """Check if any positions have hit their stop-loss. Returns symbols to close."""
        to_close = []
        for sym, pos in self._positions.items():
            price = prices.get(sym, pos.mark_price)
            if pos.stop_loss is not None:
                if pos.side == "long" and price <= pos.stop_loss:
                    to_close.append(sym)
                    log.warning("stop_loss_hit", symbol=sym, price=price, sl=pos.stop_loss)
                elif pos.side == "short" and price >= pos.stop_loss:
                    to_close.append(sym)
                    log.warning("stop_loss_hit", symbol=sym, price=price, sl=pos.stop_loss)
        return to_close

    def check_take_profits(self, prices: dict[str, float]) -> list[str]:
        """Check if any positions have hit their take-profit. Returns symbols."""
        to_tp = []
        for sym, pos in self._positions.items():
            price = prices.get(sym, pos.mark_price)
            if pos.take_profit is not None:
                if pos.side == "long" and price >= pos.take_profit:
                    to_tp.append(sym)
                elif pos.side == "short" and price <= pos.take_profit:
                    to_tp.append(sym)
        return to_tp

    def update_equity(self, equity: float) -> None:
        self._peak_equity = max(self._peak_equity, equity)

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0
        # Note: need current_equity from caller
        return 0  # computed by caller with balance + unrealized PnL


def _infer_triple_barrier_label(pos: Position, exit_price: float) -> dict | None:
    """Infer the triple-barrier label at close using the exit price and the
    extremes reached during the position. No look-ahead — uses only data
    available at close time.

    Returns None if barrier configuration is incomplete.
    """
    if pos.pt_pct is None or pos.sl_pct is None or pos.entry_price <= 0:
        return None

    sign = 1.0 if pos.side == "long" else -1.0
    signed_ret = sign * (exit_price - pos.entry_price) / pos.entry_price

    # Sample extremes from within the position's lifetime (tracked via mark_price updates).
    if pos.side == "long":
        best_ret = (pos.highest_price - pos.entry_price) / pos.entry_price
        worst_ret = (pos.lowest_price - pos.entry_price) / pos.entry_price
    else:
        best_ret = (pos.entry_price - pos.lowest_price) / pos.entry_price
        worst_ret = (pos.entry_price - pos.highest_price) / pos.entry_price

    hit_pt = best_ret >= pos.pt_pct - 1e-9
    hit_sl = worst_ret <= -pos.sl_pct + 1e-9

    # Prefer the side that the exit price actually closed at (most reliable).
    if signed_ret >= pos.pt_pct - 1e-9:
        return {"label": 1, "barrier_touched": "pt", "return_at_touch": signed_ret}
    if signed_ret <= -pos.sl_pct + 1e-9:
        return {"label": -1, "barrier_touched": "sl", "return_at_touch": signed_ret}

    # Closed inside barriers: if an extreme hit a barrier during the life of the
    # position but the exit did not, classify by which was touched first (we
    # proxy "first" by whichever extreme is a larger multiple of the barrier).
    if hit_pt and not hit_sl:
        return {"label": 1, "barrier_touched": "pt", "return_at_touch": best_ret}
    if hit_sl and not hit_pt:
        return {"label": -1, "barrier_touched": "sl", "return_at_touch": worst_ret}
    if hit_pt and hit_sl:
        # Both extremes crossed — we can't determine order without a full path.
        # Conservative label: SL (worst case), consistent with risk-averse
        # attribution.
        return {"label": -1, "barrier_touched": "sl", "return_at_touch": worst_ret}

    # Neither barrier crossed during the position — time-out.
    if pos.max_holding_s is not None and pos.duration_seconds >= pos.max_holding_s:
        return {"label": 0, "barrier_touched": "time", "return_at_touch": signed_ret}
    # Closed manually / early (LLM discretion) — record as "time" with actual return.
    return {"label": 0, "barrier_touched": "time", "return_at_touch": signed_ret}
