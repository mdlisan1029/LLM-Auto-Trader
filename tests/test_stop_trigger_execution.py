"""Regression tests for exchange-backed SL/TP triggered exits."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from src.bot import TradingBot
from src.execution.executor import ExecutionResult, TradeExecutor
from src.execution.position_manager import PositionManager


class SimpleNamespace:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


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
    exchange = Mock()
    exchange.create_order = AsyncMock(
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
    exchange = Mock()
    exchange.create_order = AsyncMock(side_effect=RuntimeError("exchange unavailable"))
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
    bot.position_manager = Mock()
    bot.position_manager.check_stop_losses.return_value = [SYMBOL_SL]
    bot.position_manager.check_take_profits.return_value = [SYMBOL_TP]
    bot.position_manager.close_position = Mock(
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

    bot.executor = SimpleNamespace(
        close_position_for_trigger=AsyncMock(side_effect=close_trigger)
    )
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.circuit_breakers = SimpleNamespace(
        record_loss=Mock(),
        _halt_trading=Mock(),
    )
    bot.symbol_cooldown = SimpleNamespace(on_stop_out=Mock())
    bot.settings = SimpleNamespace(
        afml=SimpleNamespace(enable_all=True, enable_sl_cooldown=True)
    )
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )

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
    bot.position_manager = Mock()
    bot.position_manager.check_stop_losses.return_value = [SYMBOL_SL]
    bot.position_manager.check_take_profits.return_value = []
    bot.position_manager.close_position = Mock(
        side_effect=AssertionError("Bot must not close positions locally")
    )
    bot.executor = SimpleNamespace(
        close_position_for_trigger=AsyncMock(
            return_value=ExecutionResult(
                success=False,
                action="close",
                symbol=SYMBOL_SL,
                side="",
                error="exchange unavailable",
            )
        )
    )
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.circuit_breakers = SimpleNamespace(
        record_loss=Mock(),
        _halt_trading=Mock(),
    )
    bot.symbol_cooldown = SimpleNamespace(on_stop_out=Mock())
    bot.settings = SimpleNamespace(afml=None)
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )

    await bot._process_exit_triggers({SYMBOL_SL: 97_900.0})

    bot.memory.record_trade.assert_not_called()
    bot.db.save_trade.assert_not_awaited()
    bot.circuit_breakers.record_loss.assert_not_called()
    bot.symbol_cooldown.on_stop_out.assert_not_called()
    bot.telegram.notify_trade_close.assert_not_awaited()
    bot.position_manager.close_position.assert_not_called()
    bot.circuit_breakers._halt_trading.assert_called_once_with(600)
    bot.telegram.send_circuit_breaker_alert.assert_awaited_once()
    alert = bot.telegram.send_circuit_breaker_alert.await_args.args
    assert alert[0] == "protective_exit_close_failed"
    assert alert[1] == "HALT_NEW"


SYMBOL_MARGIN = "SOL/USDT:USDT"


@pytest.mark.asyncio
async def test_margin_force_close_uses_executor_and_records_only_confirmed_trade() -> None:
    trade = {
        "symbol": SYMBOL_MARGIN,
        "side": "short",
        "entry_price": 150.0,
        "exit_price": 142.0,
        "pnl": 8.0,
        "pnl_pct": 1.6,
        "duration": "12m",
    }
    result = ExecutionResult(
        success=True,
        action="close",
        symbol=SYMBOL_MARGIN,
        side="short",
        price=142.0,
        order_id="margin-close-1",
        order_status="closed",
        filled_size=2.0,
        remaining_size=0.0,
        trade_record=trade,
    )
    bot = TradingBot.__new__(TradingBot)
    bot.executor = SimpleNamespace(
        close_position_for_trigger=AsyncMock(return_value=result)
    )
    bot.position_manager = Mock()
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )
    bot.circuit_breakers = SimpleNamespace(_halt_trading=Mock())

    await bot._process_margin_force_close(SYMBOL_MARGIN, 141.5)

    bot.executor.close_position_for_trigger.assert_awaited_once_with(
        symbol=SYMBOL_MARGIN,
        current_price=141.5,
        reason="margin_buffer_force_close",
    )
    bot.position_manager.close_position.assert_not_called()
    bot.memory.record_trade.assert_called_once_with(trade)
    bot.db.save_trade.assert_awaited_once_with(trade)
    bot.telegram.notify_trade_close.assert_awaited_once_with(
        symbol=SYMBOL_MARGIN,
        side="short",
        entry_price=150.0,
        exit_price=142.0,
        pnl=8.0,
        pnl_pct=1.6,
        duration="12m",
        reasoning="Margin liquidation-buffer force-close",
    )
    bot.telegram.send_circuit_breaker_alert.assert_not_awaited()
    bot.circuit_breakers._halt_trading.assert_not_called()


@pytest.mark.asyncio
async def test_partial_margin_force_close_does_not_record_success_and_halts_entries() -> None:
    result = ExecutionResult(
        success=False,
        action="close",
        symbol=SYMBOL_MARGIN,
        side="long",
        order_id="margin-close-partial",
        order_status="canceled",
        filled_size=0.4,
        remaining_size=0.6,
        error="Partial close fill: 0.4 of 1.0; 0.6 remains open",
    )
    bot = TradingBot.__new__(TradingBot)
    bot.executor = SimpleNamespace(
        close_position_for_trigger=AsyncMock(return_value=result)
    )
    bot.position_manager = Mock()
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )
    bot.circuit_breakers = SimpleNamespace(_halt_trading=Mock())

    await bot._process_margin_force_close(SYMBOL_MARGIN, 95.0)

    bot.position_manager.close_position.assert_not_called()
    bot.memory.record_trade.assert_not_called()
    bot.db.save_trade.assert_not_awaited()
    bot.telegram.notify_trade_close.assert_not_awaited()
    bot.circuit_breakers._halt_trading.assert_called_once_with(600)
    bot.telegram.send_circuit_breaker_alert.assert_awaited_once()
    alert_args = bot.telegram.send_circuit_breaker_alert.await_args.args
    assert alert_args[0] == "margin_force_close_failed"
    assert alert_args[1] == "HALT_NEW"
    assert "0.6" in alert_args[2]



@pytest.mark.asyncio
async def test_unknown_margin_force_close_reports_uncertain_exposure_and_longer_halt() -> None:
    result = ExecutionResult(
        success=False,
        action="close",
        symbol=SYMBOL_MARGIN,
        side="long",
        order_id="margin-close-unknown",
        order_status="open",
        filled_size=0.0,
        remaining_size=1.0,
        state_uncertain=True,
        error="Exchange order outcome is unknown; up to 1.0 may remain exposed",
    )
    bot = TradingBot.__new__(TradingBot)
    bot.executor = SimpleNamespace(
        close_position_for_trigger=AsyncMock(return_value=result)
    )
    bot.position_manager = Mock()
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )
    bot.circuit_breakers = SimpleNamespace(_halt_trading=Mock())

    await bot._process_margin_force_close(SYMBOL_MARGIN, 95.0)

    bot.circuit_breakers._halt_trading.assert_called_once_with(3600)
    bot.memory.record_trade.assert_not_called()
    bot.db.save_trade.assert_not_awaited()
    bot.telegram.notify_trade_close.assert_not_awaited()
    bot.telegram.send_circuit_breaker_alert.assert_awaited_once()
    alert = bot.telegram.send_circuit_breaker_alert.await_args.args
    assert alert[0] == "margin_force_close_failed"
    assert alert[1] == "HALT_NEW"
    assert "up to 1.0 may remain exposed" in alert[2]
    assert "actual exchange exposure is unknown" in alert[2]


@pytest.mark.asyncio
async def test_circuit_breaker_emergency_close_records_and_notifies_confirmed_trade() -> None:
    trade = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry_price": 100_000.0,
        "exit_price": 95_000.0,
        "pnl": -50.0,
        "pnl_pct": -10.0,
        "duration": "30m",
    }
    result = ExecutionResult(
        success=True,
        action="emergency_close",
        symbol=trade["symbol"],
        side="long",
        price=94_950.0,
        order_id="emergency-close-1",
        order_status="closed",
        filled_size=0.01,
        remaining_size=0.0,
        trade_record=trade,
    )
    bot = TradingBot.__new__(TradingBot)
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )
    bot.circuit_breakers = SimpleNamespace(_halt_trading=Mock())

    await bot._process_emergency_close_results([result], "portfolio drawdown")

    bot.memory.record_trade.assert_called_once_with(trade)
    bot.db.save_trade.assert_awaited_once_with(trade)
    bot.telegram.notify_trade_close.assert_awaited_once_with(
        symbol="BTC/USDT:USDT",
        side="long",
        entry_price=100_000.0,
        exit_price=94_950.0,
        pnl=-50.0,
        pnl_pct=-10.0,
        duration="30m",
        reasoning="Circuit-breaker emergency close: portfolio drawdown",
    )
    bot.telegram.send_circuit_breaker_alert.assert_not_awaited()
    bot.circuit_breakers._halt_trading.assert_not_called()


@pytest.mark.asyncio
async def test_partial_emergency_close_is_alerted_and_not_recorded_as_completed() -> None:
    result = ExecutionResult(
        success=False,
        action="emergency_close",
        symbol="ETH/USDT:USDT",
        side="short",
        order_id="emergency-close-partial",
        order_status="canceled",
        filled_size=0.25,
        remaining_size=0.75,
        error="Emergency close partially filled: 0.25 of 1.0; 0.75 remains open",
    )
    bot = TradingBot.__new__(TradingBot)
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )
    bot.circuit_breakers = SimpleNamespace(_halt_trading=Mock())

    await bot._process_emergency_close_results([result], "websocket disconnected")

    bot.memory.record_trade.assert_not_called()
    bot.db.save_trade.assert_not_awaited()
    bot.telegram.notify_trade_close.assert_not_awaited()
    bot.circuit_breakers._halt_trading.assert_called_once_with(3600)
    bot.telegram.send_circuit_breaker_alert.assert_awaited_once()
    alert_args = bot.telegram.send_circuit_breaker_alert.await_args.args
    assert alert_args[0] == "emergency_close_incomplete"
    assert alert_args[1] == "HALT_NEW"
    assert "0.75" in alert_args[2]


@pytest.mark.asyncio
async def test_emergency_close_batch_continues_and_flags_missing_trade_record() -> None:
    trade = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry_price": 100_000.0,
        "exit_price": 99_500.0,
        "pnl": -5.0,
        "pnl_pct": -1.0,
    }
    confirmed = ExecutionResult(
        success=True,
        action="emergency_close",
        symbol=trade["symbol"],
        side="long",
        price=99_500.0,
        order_status="closed",
        filled_size=0.01,
        remaining_size=0.0,
        trade_record=trade,
    )
    unaccounted = ExecutionResult(
        success=True,
        action="emergency_close",
        symbol="ETH/USDT:USDT",
        side="short",
        order_id="emergency-close-no-record",
        order_status="closed",
        filled_size=1.0,
        remaining_size=0.0,
        trade_record=None,
    )
    bot = TradingBot.__new__(TradingBot)
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )
    bot.circuit_breakers = SimpleNamespace(_halt_trading=Mock())

    await bot._process_emergency_close_results(
        [confirmed, unaccounted], "portfolio drawdown",
    )

    bot.memory.record_trade.assert_called_once_with(trade)
    bot.db.save_trade.assert_awaited_once_with(trade)
    bot.telegram.notify_trade_close.assert_awaited_once()
    bot.circuit_breakers._halt_trading.assert_called_once_with(3600)
    bot.telegram.send_circuit_breaker_alert.assert_awaited_once()
    alert = bot.telegram.send_circuit_breaker_alert.await_args.args
    assert alert[0] == "emergency_close_incomplete"
    assert alert[1] == "HALT_NEW"
    assert "without a completed trade record" in alert[2]
    assert "Trigger: portfolio drawdown" in alert[2]


@pytest.mark.asyncio
async def test_emergency_close_reporting_errors_halt_and_alert_after_all_attempts() -> None:
    trade = {
        "symbol": "BTC/USDT:USDT",
        "side": "long",
        "entry_price": 100_000.0,
        "exit_price": 99_500.0,
        "pnl": -5.0,
        "pnl_pct": -1.0,
    }
    result = ExecutionResult(
        success=True,
        action="emergency_close",
        symbol=trade["symbol"],
        side="long",
        price=99_500.0,
        order_status="closed",
        filled_size=0.01,
        remaining_size=0.0,
        trade_record=trade,
    )
    bot = TradingBot.__new__(TradingBot)
    bot.memory = SimpleNamespace(
        record_trade=Mock(side_effect=RuntimeError("memory unavailable"))
    )
    bot.db = SimpleNamespace(
        save_trade=AsyncMock(side_effect=RuntimeError("database unavailable"))
    )
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(side_effect=RuntimeError("telegram unavailable")),
        send_circuit_breaker_alert=AsyncMock(),
    )
    bot.circuit_breakers = SimpleNamespace(_halt_trading=Mock())

    await bot._process_emergency_close_results([result], "drawdown")

    bot.memory.record_trade.assert_called_once_with(trade)
    bot.db.save_trade.assert_awaited_once_with(trade)
    bot.telegram.notify_trade_close.assert_awaited_once()
    bot.circuit_breakers._halt_trading.assert_called_once_with(3600)
    bot.telegram.send_circuit_breaker_alert.assert_awaited_once()
    alert = bot.telegram.send_circuit_breaker_alert.await_args.args
    assert alert[0] == "emergency_close_reporting_failed"
    assert alert[1] == "HALT_NEW"
    assert "memory: memory unavailable" in alert[2]
    assert "database: database unavailable" in alert[2]
    assert "telegram: telegram unavailable" in alert[2]


@pytest.mark.asyncio
async def test_unknown_emergency_close_reports_max_possible_exposure() -> None:
    result = ExecutionResult(
        success=False,
        action="emergency_close",
        symbol="ETH/USDT:USDT",
        side="short",
        order_id="emergency-close-unknown",
        order_status="open",
        filled_size=0.0,
        remaining_size=0.75,
        state_uncertain=True,
        error="Exchange order outcome is unknown; up to 0.75 may remain exposed",
    )
    bot = TradingBot.__new__(TradingBot)
    bot.memory = SimpleNamespace(record_trade=Mock())
    bot.db = SimpleNamespace(save_trade=AsyncMock())
    bot.telegram = SimpleNamespace(
        notify_trade_close=AsyncMock(),
        send_circuit_breaker_alert=AsyncMock(),
    )
    bot.circuit_breakers = SimpleNamespace(_halt_trading=Mock())

    await bot._process_emergency_close_results([result], "exchange timeout")

    bot.circuit_breakers._halt_trading.assert_called_once_with(3600)
    bot.memory.record_trade.assert_not_called()
    bot.db.save_trade.assert_not_awaited()
    bot.telegram.notify_trade_close.assert_not_awaited()
    bot.telegram.send_circuit_breaker_alert.assert_awaited_once()
    alert = bot.telegram.send_circuit_breaker_alert.await_args.args
    assert alert[0] == "emergency_close_incomplete"
    assert alert[1] == "HALT_NEW"
    assert "up to 0.75 may remain exposed" in alert[2]
    assert "actual exchange exposure is unknown" in alert[2]
