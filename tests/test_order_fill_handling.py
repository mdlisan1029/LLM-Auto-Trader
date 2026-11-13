"""Regression tests for partial and unfilled exchange order handling."""

from __future__ import annotations

import pytest

from src.execution.executor import TradeExecutor
from src.execution.position_manager import PositionManager
from src.llm.response_parser import ActionType, TradeAction
from src.risk.risk_manager import ValidationResult


SYMBOL = "BTC/USDT:USDT"


class StubExchange:
    def __init__(
        self,
        *,
        create_orders: list[dict],
        cancel_response: dict | None = None,
        fetch_response: dict | None = None,
        precision: dict | None = None,
    ) -> None:
        self._create_orders = list(create_orders)
        self._cancel_response = cancel_response
        self._fetch_response = fetch_response
        self._precision = precision or {"amount": 0.001, "price": 0.01}
        self.create_calls: list[tuple] = []
        self.cancel_calls: list[tuple] = []
        self.fetch_calls: list[tuple] = []
        self.leverage_calls: list[tuple] = []

    def market(self, symbol: str) -> dict:
        return {"precision": self._precision}

    async def set_leverage(self, leverage: int, symbol: str) -> None:
        self.leverage_calls.append((leverage, symbol))

    async def create_order(self, *args: object, **kwargs: object) -> dict:
        self.create_calls.append((args, kwargs))
        if not self._create_orders:
            raise AssertionError("Unexpected create_order call")
        return self._create_orders.pop(0)

    async def cancel_order(self, order_id: str, symbol: str) -> dict | None:
        self.cancel_calls.append((order_id, symbol))
        return self._cancel_response

    async def fetch_order(self, order_id: str, symbol: str) -> dict | None:
        self.fetch_calls.append((order_id, symbol))
        return self._fetch_response


def _open_position(position_manager: PositionManager) -> None:
    position_manager.open_position(
        symbol=SYMBOL,
        side="long",
        size=0.01,
        notional=1_000.0,
        entry_price=100_000.0,
        leverage=2,
        stop_loss=98_000.0,
        take_profit=104_000.0,
        fees=0.2,
        reasoning="test position",
    )


@pytest.mark.asyncio
async def test_partial_close_cancels_remainder_and_tracks_only_confirmed_fill() -> None:
    position_manager = PositionManager()
    _open_position(position_manager)
    terminal_partial = {
        "id": "close-1",
        "status": "canceled",
        "amount": 0.01,
        "filled": 0.004,
        "remaining": 0.006,
        "average": 99_000.0,
        "fee": {"cost": 0.04},
    }
    exchange = StubExchange(
        create_orders=[{
            "id": "close-1",
            "status": "open",
            "amount": 0.01,
            "filled": 0.004,
            "remaining": 0.006,
            "average": 99_000.0,
        }],
        cancel_response=terminal_partial,
        fetch_response=terminal_partial,
    )
    executor = TradeExecutor(exchange, position_manager, paper_mode=False)

    result = await executor.close_position_for_trigger(
        symbol=SYMBOL,
        current_price=98_900.0,
        reason="stop_loss",
    )

    position = position_manager.get_position(SYMBOL)
    assert not result.success
    assert result.order_status == "canceled"
    assert result.filled_size == pytest.approx(0.004)
    assert result.remaining_size == pytest.approx(0.006)
    assert result.trade_record is None
    assert position is not None
    assert position.size == pytest.approx(0.006)
    assert position.notional == pytest.approx(600.0)
    assert position.realized_pnl == pytest.approx(-4.0)
    assert position.fees_paid == pytest.approx(0.24)
    assert exchange.cancel_calls == [("close-1", SYMBOL)]
    assert exchange.fetch_calls == [("close-1", SYMBOL)]


@pytest.mark.asyncio
async def test_unfilled_close_keeps_entire_local_position_open() -> None:
    position_manager = PositionManager()
    _open_position(position_manager)
    exchange = StubExchange(create_orders=[{
        "id": "close-empty",
        "status": "canceled",
        "amount": 0.01,
        "filled": 0.0,
        "remaining": 0.01,
    }])
    executor = TradeExecutor(exchange, position_manager, paper_mode=False)

    result = await executor.close_position_for_trigger(
        symbol=SYMBOL,
        current_price=97_900.0,
        reason="stop_loss",
    )

    position = position_manager.get_position(SYMBOL)
    assert not result.success
    assert result.filled_size == 0
    assert result.remaining_size == pytest.approx(0.01)
    assert position is not None
    assert position.size == pytest.approx(0.01)
    assert position.notional == pytest.approx(1_000.0)
    assert position.realized_pnl == 0


@pytest.mark.asyncio
async def test_ambiguous_order_without_terminal_status_is_not_assumed_filled() -> None:
    position_manager = PositionManager()
    _open_position(position_manager)
    exchange = StubExchange(
        create_orders=[{
            "id": "close-unknown",
            "amount": 0.01,
            "filled": 0.01,
        }]
    )
    executor = TradeExecutor(exchange, position_manager, paper_mode=False)

    result = await executor.close_position_for_trigger(
        symbol=SYMBOL,
        current_price=97_900.0,
        reason="stop_loss",
    )

    position = position_manager.get_position(SYMBOL)
    assert not result.success
    assert "terminal state" in result.error
    assert position is not None
    assert position.size == pytest.approx(0.01)
    assert exchange.fetch_calls == [("close-unknown", SYMBOL)]


@pytest.mark.asyncio
async def test_close_response_without_filled_quantity_does_not_use_requested_size() -> None:
    position_manager = PositionManager()
    _open_position(position_manager)
    exchange = StubExchange(create_orders=[{
        "id": "close-no-fill-field",
        "status": "canceled",
        "amount": 0.01,
        "remaining": 0.01,
    }])
    executor = TradeExecutor(exchange, position_manager, paper_mode=False)

    result = await executor.close_position_for_trigger(
        symbol=SYMBOL,
        current_price=97_900.0,
        reason="stop_loss",
    )

    position = position_manager.get_position(SYMBOL)
    assert not result.success
    assert "no reliable filled quantity" in result.error
    assert position is not None
    assert position.size == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_two_stage_partial_close_preserves_realized_pnl_and_final_trade_record() -> None:
    position_manager = PositionManager()
    _open_position(position_manager)
    exchange = StubExchange(create_orders=[
        {
            "id": "partial-close",
            "status": "canceled",
            "amount": 0.01,
            "filled": 0.004,
            "remaining": 0.006,
            "average": 99_000.0,
            "fee": {"cost": 0.04},
        },
        {
            "id": "final-close",
            "status": "closed",
            "amount": 0.006,
            "filled": 0.006,
            "remaining": 0.0,
            "average": 98_000.0,
            "fee": {"cost": 0.06},
        },
    ])
    executor = TradeExecutor(exchange, position_manager, paper_mode=False)

    partial = await executor.close_position_for_trigger(
        symbol=SYMBOL,
        current_price=99_000.0,
        reason="stop_loss",
    )
    final = await executor.close_position_for_trigger(
        symbol=SYMBOL,
        current_price=98_000.0,
        reason="stop_loss",
    )

    assert not partial.success
    assert final.success
    assert final.trade_record is not None
    assert final.trade_record["size"] == pytest.approx(0.006)
    # Gross PnL: -4 from the first 40% close and -12 from the final 60%;
    # fees: 0.20 entry + 0.04 partial exit + 0.06 final exit.
    assert final.trade_record["pnl"] == pytest.approx(-16.3)
    assert position_manager.get_position(SYMBOL) is None


@pytest.mark.asyncio
async def test_entry_with_zero_confirmed_fill_does_not_open_local_position() -> None:
    action = TradeAction(
        action=ActionType.OPEN_LONG,
        symbol=SYMBOL,
        size_pct=10,
        leverage=2,
        stop_loss=98_000.0,
        take_profit=104_000.0,
        reasoning="test entry order",
    )
    validation = ValidationResult(approved=True, action=action, adjusted_size_pct=10)
    position_manager = PositionManager()
    exchange = StubExchange(create_orders=[{
        "id": "entry-empty",
        "status": "canceled",
        "amount": 0.001,
        "filled": 0.0,
        "remaining": 0.001,
    }])
    executor = TradeExecutor(exchange, position_manager, paper_mode=False)

    result = await executor.execute(
        action=action,
        validation=validation,
        current_price=100_000.0,
        equity=1_000.0,
    )

    assert not result.success
    assert result.filled_size == 0
    assert position_manager.get_position(SYMBOL) is None


@pytest.mark.asyncio
async def test_partial_entry_tracks_only_the_actual_filled_size() -> None:
    action = TradeAction(
        action=ActionType.OPEN_LONG,
        symbol=SYMBOL,
        size_pct=10,
        leverage=2,
        stop_loss=98_000.0,
        take_profit=104_000.0,
        reasoning="test partial entry",
    )
    validation = ValidationResult(approved=True, action=action, adjusted_size_pct=10)
    position_manager = PositionManager()
    exchange = StubExchange(create_orders=[{
        "id": "entry-partial",
        "status": "canceled",
        "amount": 0.001,
        "filled": 0.0005,
        "remaining": 0.0005,
        "average": 100_100.0,
    }])
    executor = TradeExecutor(exchange, position_manager, paper_mode=False)

    result = await executor.execute(
        action=action,
        validation=validation,
        current_price=100_000.0,
        equity=1_000.0,
    )

    position = position_manager.get_position(SYMBOL)
    assert result.success
    assert result.filled_size == pytest.approx(0.0005)
    assert result.remaining_size == pytest.approx(0.0005)
    assert position is not None
    assert position.size == pytest.approx(0.0005)
    assert position.entry_price == pytest.approx(100_100.0)


@pytest.mark.asyncio
async def test_partial_reduce_updates_only_confirmed_quantity() -> None:
    position_manager = PositionManager()
    _open_position(position_manager)
    exchange = StubExchange(
        create_orders=[{
            "id": "reduce-partial",
            "status": "canceled",
            "amount": 0.001,
            "filled": 0.0005,
            "remaining": 0.0005,
            "average": 99_000.0,
        }],
        precision={"amount": 0.001},
    )
    executor = TradeExecutor(exchange, position_manager, paper_mode=False)
    action = TradeAction(
        action=ActionType.REDUCE_POSITION,
        symbol=SYMBOL,
        size_pct=10,
        reasoning="test reduce",
    )

    result = await executor._reduce_position(action, current_price=99_000.0)

    position = position_manager.get_position(SYMBOL)
    assert not result.success
    assert result.filled_size == pytest.approx(0.0005)
    assert result.remaining_size == pytest.approx(0.0095)
    assert position is not None
    assert position.size == pytest.approx(0.0095)
    assert position.realized_pnl == pytest.approx(-0.5)


@pytest.mark.asyncio
async def test_partial_emergency_close_keeps_unfilled_remainder_tracked() -> None:
    position_manager = PositionManager()
    _open_position(position_manager)
    exchange = StubExchange(create_orders=[{
        "id": "emergency-partial",
        "status": "canceled",
        "amount": 0.01,
        "filled": 0.004,
        "remaining": 0.006,
        "average": 97_000.0,
    }])
    executor = TradeExecutor(exchange, position_manager, paper_mode=False)

    results = await executor.emergency_close_all()

    position = position_manager.get_position(SYMBOL)
    assert len(results) == 1
    assert not results[0].success
    assert results[0].filled_size == pytest.approx(0.004)
    assert results[0].remaining_size == pytest.approx(0.006)
    assert position is not None
    assert position.size == pytest.approx(0.006)
