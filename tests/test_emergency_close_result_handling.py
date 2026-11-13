"""Regression tests for circuit-breaker emergency close accounting."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.bot import TradingBot
from src.execution.executor import ExecutionResult


SYMBOL_OK = "BTC/USDT:USDT"
SYMBOL_PARTIAL = "ETH/USDT:USDT"


def _make_bot() -> TradingBot:
    bot = TradingBot.__new__(TradingBot)
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )
    bot.circuit_breakers = SimpleNamespace(_halt_trading=Mock())
    return bot


@pytest.mark.asyncio
async def test_confirmed_emergency_closes_are_saved_and_notified() -> None:
    bot = _make_bot()
    trade = {
        "symbol": SYMBOL_OK,
        "side": "long",
        "entry_price": 100_000.0,
        "exit_price": 99_000.0,
        "pnl": -10.4,
        "pnl_pct": -2.08,
        "duration": "14m",
    }
    result = ExecutionResult(
        success=True,
        action="emergency_close",
        symbol=SYMBOL_OK,
        side="long",
        size=0.01,
        price=99_000.0,
        order_id="emergency-1",
        order_status="closed",
        filled_size=0.01,
        remaining_size=0.0,
        trade_record=trade,
    )

    await bot._process_emergency_close_results([result])

    bot.memory.record_trade.assert_called_once_with(trade)
    bot.db.save_trade.assert_awaited_once_with(trade)
    bot.telegram.notify_trade_close.assert_awaited_once_with(
        symbol=SYMBOL_OK,
        side="long",
        entry_price=100_000.0,
        exit_price=99_000.0,
        pnl=-10.4,
        pnl_pct=-2.08,
        duration="14m",
        reasoning="Emergency circuit-breaker close",
    )
    bot.telegram.send_circuit_breaker_alert.assert_not_awaited()
    bot.circuit_breakers._halt_trading.assert_not_called()


@pytest.mark.asyncio
async def test_partial_emergency_close_is_not_recorded_as_completed_trade() -> None:
    bot = _make_bot()
    result = ExecutionResult(
        success=False,
        action="emergency_close",
        symbol=SYMBOL_PARTIAL,
        side="short",
        order_id="emergency-partial-1",
        order_status="canceled",
        filled_size=0.4,
        remaining_size=0.6,
        error="Emergency close partially filled: 0.4 of 1.0; 0.6 remains open",
    )

    await bot._process_emergency_close_results([result])

    bot.memory.record_trade.assert_not_called()
    bot.db.save_trade.assert_not_awaited()
    bot.telegram.notify_trade_close.assert_not_awaited()
    bot.circuit_breakers._halt_trading.assert_called_once_with(3600)
    bot.telegram.send_circuit_breaker_alert.assert_awaited_once()
    alert = bot.telegram.send_circuit_breaker_alert.await_args.args
    assert alert[0] == "emergency_close_failed"
    assert alert[1] == "HALT_NEW"
    assert SYMBOL_PARTIAL in alert[2]
    assert "Filled: 0.4" in alert[2]
    assert "remaining open: 0.6" in alert[2]


@pytest.mark.asyncio
async def test_emergency_close_batch_processes_success_and_reports_missing_trade_record() -> None:
    bot = _make_bot()
    trade = {
        "symbol": SYMBOL_OK,
        "side": "long",
        "entry_price": 100_000.0,
        "exit_price": 99_500.0,
        "pnl": -5.0,
        "pnl_pct": -1.0,
    }
    confirmed = ExecutionResult(
        success=True,
        action="emergency_close",
        symbol=SYMBOL_OK,
        side="long",
        price=99_500.0,
        order_status="closed",
        filled_size=0.01,
        remaining_size=0.0,
        trade_record=trade,
    )
    no_record = ExecutionResult(
        success=True,
        action="emergency_close",
        symbol=SYMBOL_PARTIAL,
        side="short",
        order_id="emergency-2",
        order_status="closed",
        filled_size=1.0,
        remaining_size=0.0,
        error="",
        trade_record=None,
    )

    await bot._process_emergency_close_results([confirmed, no_record])

    bot.memory.record_trade.assert_called_once_with(trade)
    bot.db.save_trade.assert_awaited_once_with(trade)
    bot.telegram.notify_trade_close.assert_awaited_once()
    bot.circuit_breakers._halt_trading.assert_called_once_with(3600)
    bot.telegram.send_circuit_breaker_alert.assert_awaited_once()
    alert = bot.telegram.send_circuit_breaker_alert.await_args.args
    assert alert[0] == "emergency_close_failed"
    assert "without a completed trade record" in alert[2]
