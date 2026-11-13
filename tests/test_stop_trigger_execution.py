"""Regression tests for exchange-backed SL/TP triggered exits."""

from __future__ import annotations

import types
import unittest.mock

import pytest

from src.bot import TradingBot
from src.execution.executor import ExecutionResult, TradeExecutor
from src.execution.position_manager import PositionManager


SYMBOL_SL = "BTC/USDT:USDT"
SYMBOL_TP = "ETH/USDT:USDT"


@pytest.mark.asyncio
async def test_live_triggered_close_submits_reduce_only_order_and_records_trade() -> None:
    position_manager = PositionManager()
    position_manager.open_position(
        symbol=SYMBOL_SL,
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
    exchange = unittest.mock.Mock()
    exchange.create_order = unittest.mock.AsyncMock(
        return_value={
            "id": "close-order-1",
            "status": "closed",
            "amount": 0.01,
            "filled": 0.01,
            "average": 97_900.0,
            "fee": {"cost": 0.04},
        }
    )
    executor = TradeExecutor(
        exchange=exchange,
        position_manager=position_manager,
        paper_mode=False,
        taker_fee=0.0004,
    )

    result = await executor.close_position_for_trigger(
        symbol=SYMBOL_SL,
        current_price=97_900.0,
        reason="stop_loss",
    )

    assert result.success
    assert result.order_id == "close-order-1"
    assert result.trade_record is not None
    assert result.trade_record["exit_price"] == 97_900.0
    assert position_manager.get_position(SYMBOL_SL) is None
    exchange.create_order.assert_awaited_once_with(
        SYMBOL_SL,
        "market",
        "sell",
        0.01,
        params={"reduceOnly": True},
    )


@pytest.mark.asyncio
async def test_triggered_close_failure_keeps_local_position_open() -> None:
    position_manager = PositionManager()
    position_manager.open_position(
        symbol=SYMBOL_SL,
        side="long",
        size=0.01,
        notional=1_000.0,
        entry_price=100_000.0,
        leverage=2,
        stop_loss=98_000.0,
        take_profit=104_000.0,
        reasoning="test position",
    )
    exchange = unittest.mock.Mock()
    exchange.create_order = unittest.mock.AsyncMock(side_effect=RuntimeError("exchange unavailable"))
    executor = TradeExecutor(
        exchange=exchange,
        position_manager=position_manager,
        paper_mode=False,
    )

    result = await executor.close_position_for_trigger(
        symbol=SYMBOL_SL,
        current_price=97_900.0,
        reason="stop_loss",
    )

    assert not result.success
    assert "exchange unavailable" in result.error
    assert position_manager.get_position(SYMBOL_SL) is not None


@pytest.mark.asyncio
async def test_bot_routes_stop_loss_and_take_profit_through_executor() -> None:
    bot = TradingBot.__new__(TradingBot)
    bot.position_manager = unittest.mock.Mock()
    bot.position_manager.check_stop_losses.return_value = [SYMBOL_SL]
    bot.position_manager.check_take_profits.return_value = [SYMBOL_TP]
    bot.position_manager.close_position = unittest.mock.Mock(
        side_effect=AssertionError("Bot must not close positions locally")
    )

    async def close_trigger(*, symbol: str, current_price: float, reason: str):
        trade = {
            "symbol": symbol,
            "side": "long",
            "entry_price": current_price - 10,
            "exit_price": current_price,
            "pnl": -1.0 if reason == "stop_loss" else 1.0,
            "pnl_pct": -0.5 if reason == "stop_loss" else 0.5,
            "duration": "5m",
        }
        return ExecutionResult(
            success=True,
            action="close",
            symbol=symbol,
            side="long",
            price=current_price,
            order_id=f"{reason}-order",
            trade_record=trade,
        )

    bot.executor = types.SimpleNamespace(
        close_position_for_trigger=unittest.mock.AsyncMock(side_effect=close_trigger)
    )
    bot.memory = types.SimpleNamespace(record_trade=unittest.mock.Mock())
    bot.db = types.SimpleNamespace(save_trade=unittest.mock.AsyncMock())
    bot.circuit_breakers = types.SimpleNamespace(record_loss=unittest.mock.Mock())
    bot.symbol_cooldown = types.SimpleNamespace(on_stop_out=unittest.mock.Mock())
    bot.settings = types.SimpleNamespace(
        afml=types.SimpleNamespace(enable_all=True, enable_sl_cooldown=True)
    )
    bot.telegram = types.SimpleNamespace(notify_trade_close=unittest.mock.AsyncMock())

    await bot._process_exit_triggers(
        {SYMBOL_SL: 97_900.0, SYMBOL_TP: 3_500.0}
    )

    assert bot.executor.close_position_for_trigger.await_count == 2
    bot.executor.close_position_for_trigger.assert_any_await(
        symbol=SYMBOL_SL,
        current_price=97_900.0,
        reason="stop_loss",
    )
    bot.executor.close_position_for_trigger.assert_any_await(
        symbol=SYMBOL_TP,
        current_price=3_500.0,
        reason="take_profit",
    )
    assert bot.memory.record_trade.call_count == 2
    assert bot.db.save_trade.await_count == 2
    assert bot.telegram.notify_trade_close.await_count == 2
    bot.circuit_breakers.record_loss.assert_called_once_with(-0.005)
    bot.symbol_cooldown.on_stop_out.assert_called_once()
    bot.position_manager.close_position.assert_not_called()


@pytest.mark.asyncio
async def test_bot_does_not_record_or_notify_when_exchange_close_fails() -> None:
    bot = TradingBot.__new__(TradingBot)
    bot.position_manager = unittest.mock.Mock()
    bot.position_manager.check_stop_losses.return_value = [SYMBOL_SL]
    bot.position_manager.check_take_profits.return_value = []
    bot.position_manager.close_position = unittest.mock.Mock(
        side_effect=AssertionError("Bot must not close positions locally")
    )
    bot.executor = types.SimpleNamespace(
        close_position_for_trigger=unittest.mock.AsyncMock(
            return_value=ExecutionResult(
                success=False,
                action="close",
                symbol=SYMBOL_SL,
                side="",
                error="exchange unavailable",
            )
        )
    )
    bot.memory = types.SimpleNamespace(record_trade=unittest.mock.Mock())
    bot.db = types.SimpleNamespace(save_trade=unittest.mock.AsyncMock())
    bot.circuit_breakers = types.SimpleNamespace(record_loss=unittest.mock.Mock())
    bot.symbol_cooldown = types.SimpleNamespace(on_stop_out=unittest.mock.Mock())
    bot.settings = types.SimpleNamespace(afml=None)
    bot.telegram = types.SimpleNamespace(notify_trade_close=unittest.mock.AsyncMock())

    await bot._process_exit_triggers({SYMBOL_SL: 97_900.0})

    bot.memory.record_trade.assert_not_called()
    bot.db.save_trade.assert_not_awaited()
    bot.circuit_breakers.record_loss.assert_not_called()
    bot.symbol_cooldown.on_stop_out.assert_not_called()
    bot.telegram.notify_trade_close.assert_not_awaited()
    bot.position_manager.close_position.assert_not_called()
