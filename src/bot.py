"""Main trading bot orchestrator."""

from __future__ import annotations

import asyncio
import signal
import time
from datetime import datetime, timezone

import aiohttp

from config.settings import Settings
from config.symbols import get_enabled_symbols
from src.analysis.cusum import CUSUMFilter
from src.analysis.market_state import MarketState
from src.data.aggregator import DataAggregator
from src.data.events import EventCalendar
from src.data.macro import MacroCollector
from src.data.market import MarketDataCollector
from src.data.news import NewsCollector
from src.data.onchain import OnChainCollector
from src.data.prediction_markets import PredictionMarketsCollector
from src.data.sentiment import SentimentCollector
from src.data.stablecoin import USDTMonitor
from src.data.top_traders import TopTraderCollector
from src.data.websocket import WebSocketManager
from src.execution.executor import TradeExecutor
from src.execution.position_manager import PositionManager
from src.llm.client import TradingLLMClient
from src.llm.cost_tracker import CostTracker
from src.llm.memory import TradingMemory
from src.llm.response_parser import ActionType
from src.llm.scheduler import LLMScheduler
from src.risk.circuit_breakers import BreakerAction, CircuitBreakers
from src.risk.cooldown import SymbolCooldown
from src.risk.daily_peak import DailyPeakTracker
from src.risk.margin_monitor import compute_margin_state
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager
from src.storage.database import Database
from src.storage.retention import RetentionManager
from src.telegram.reporter import TelegramReporter
from src.utils.errors import LLMBudgetExceededError
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


class TradingBot:
    """Main orchestrator that ties all components together."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._running = False
        self._shutdown_event = asyncio.Event()

        # --- Initialize components ---

        # Shared HTTP session for all collectors (connection pooling)
        self._http_session: aiohttp.ClientSession | None = None

        # Data
        self.market = MarketDataCollector(
            trading_exchange_id=settings.exchange.name,
            trading_api_key=settings.exchange.api_key,
            trading_secret=settings.exchange.secret,
            data_exchange_ids=settings.exchange.data_exchanges,
            market_type=settings.exchange.market_type,
            quote_currency=settings.exchange.quote_currency,
        )
        self.websocket = WebSocketManager(
            exchange_id=settings.exchange.name,
            api_key=settings.exchange.api_key,
            secret=settings.exchange.secret,
            market_type=settings.exchange.market_type,
        )
        self.onchain = OnChainCollector()
        self.sentiment = SentimentCollector()
        self.macro = MacroCollector()
        # New data sources (initialized with shared session in initialize())
        self.prediction_markets: PredictionMarketsCollector | None = None
        self.top_traders: TopTraderCollector | None = None
        self.news: NewsCollector | None = None

        # Position & Risk
        self.position_manager = PositionManager()
        self.position_sizer = PositionSizer(
            taker_fee=settings.exchange.taker_fee,
        )
        self.circuit_breakers = CircuitBreakers(
            max_drawdown_pct=settings.risk.max_drawdown_pct,
            drawdown_warning_pct=settings.risk.drawdown_warning_pct,
            daily_loss_limit_pct=settings.risk.daily_loss_limit_pct,
            rapid_loss_count=settings.risk.rapid_loss_count,
            rapid_loss_pct=settings.risk.rapid_loss_pct,
            rapid_loss_window_seconds=settings.risk.rapid_loss_window_seconds,
            volatility_spike_multiplier=settings.risk.volatility_spike_multiplier,
            max_spread_pct=settings.risk.max_spread_pct,
            max_funding_rate_8h=settings.risk.max_funding_rate_8h,
            ws_disconnect_seconds=settings.risk.ws_disconnect_close_seconds,
            llm_failure_threshold=settings.risk.llm_failure_threshold,
        )
        # Phase 2 — cross-symbol market state (regime, baselines, GARCH)
        self.market_state = MarketState(afml_settings=settings.afml)
        # Phase 3 risk components (must be created BEFORE RiskManager that references them)
        self.symbol_cooldown = SymbolCooldown(
            base_minutes=float(settings.afml.sl_cooldown_base_minutes),
            chop_count=int(settings.afml.chop_sl_count),
            chop_window_hours=float(settings.afml.chop_window_hours),
            chop_lockout_hours=float(settings.afml.chop_lockout_hours),
        )
        self.daily_peak = DailyPeakTracker()
        self.usdt_monitor = USDTMonitor(
            threshold=float(settings.afml.usdt_depeg_threshold),
            sustain_count=int(settings.afml.usdt_depeg_sustain_count),
        )
        self.event_calendar = EventCalendar(
            pre_halt_minutes=int(settings.afml.event_pre_halt_minutes),
            warning_hours=int(settings.afml.event_warning_hours),
        )
        self._latest_margin_state = None

        self.risk_manager = RiskManager(
            max_leverage=settings.risk.max_leverage,
            max_position_pct=settings.risk.max_position_pct,
            max_total_exposure_pct=settings.risk.max_total_exposure_pct,
            max_risk_per_trade_pct=settings.risk.max_risk_per_trade_pct,
            min_reward_risk_ratio=settings.risk.min_reward_risk_ratio,
            max_concurrent_positions=settings.risk.max_concurrent_positions,
            position_sizer=self.position_sizer,
            circuit_breakers=self.circuit_breakers,
            afml_settings=settings.afml,
            market_state=self.market_state,
            symbol_cooldown=self.symbol_cooldown,
            event_calendar=self.event_calendar,
        )

        # LLM provider — built once, shared with the cost tracker so both
        # see the same pricing table.
        from src.llm.providers.factory import build_provider
        self.llm_provider = build_provider(settings.llm)

        # User-supplied price overrides take precedence over the provider's
        # built-in table. A price of 0 means "use the provider default".
        if settings.llm.input_price_per_m > 0:
            self.cost_tracker = CostTracker(
                daily_budget=settings.llm.daily_budget_usd,
                monthly_budget=settings.llm.monthly_budget_usd,
                input_price_per_m=settings.llm.input_price_per_m,
                cached_price_per_m=settings.llm.cached_input_price_per_m,
                output_price_per_m=settings.llm.output_price_per_m,
            )
        else:
            self.cost_tracker = CostTracker.from_provider(
                self.llm_provider,
                daily_budget=settings.llm.daily_budget_usd,
                monthly_budget=settings.llm.monthly_budget_usd,
            )
        self.memory = TradingMemory()
        # Per-symbol CUSUM event filters (Phase 1 §3.2 — event-based LLM trigger)
        self._cusum_filters: dict[str, CUSUMFilter] = {}
        self._last_cusum_price: dict[str, float] = {}
        self._last_cusum_event_time: float = 0.0
        self.scheduler = LLMScheduler(
            base_interval=settings.llm.base_interval,
            min_interval=settings.llm.min_interval,
            max_interval=settings.llm.max_interval,
            default_no_position=settings.llm.default_no_position,
        )
        from src.llm.client import render_system_prompt_for
        self.llm_client = TradingLLMClient(
            provider=self.llm_provider,
            max_output_tokens=settings.llm.max_output_tokens,
            temperature=settings.llm.temperature,
            max_retries=settings.llm.max_retries,
            cost_tracker=self.cost_tracker,
            memory=self.memory,
            scheduler=self.scheduler,
            afml_settings=settings.afml,
            market_state=self.market_state,
            symbol_cooldown=self.symbol_cooldown,
            event_calendar=self.event_calendar,
            usdt_monitor=self.usdt_monitor,
            daily_peak=self.daily_peak,
            margin_monitor=lambda: self._latest_margin_state,
            system_prompt=render_system_prompt_for(settings),
        )

        # Storage
        self.db = Database(settings.storage.db_path)

        # Telegram
        self.telegram = TelegramReporter(
            bot_token=settings.telegram.bot_token,
            chat_id=settings.telegram.chat_id,
            min_interval_seconds=settings.telegram.min_notification_interval_seconds,
        )

        # Symbols and aggregator (set during initialize)
        self._symbols: list[str] = []
        self.aggregator: DataAggregator | None = None
        self.executor: TradeExecutor | None = None

        # Timing
        self._last_onchain_fetch: float = 0
        self._last_macro_fetch: float = 0
        self._last_prediction_markets_fetch: float = 0
        self._last_top_traders_fetch: float = 0
        self._last_news_fetch: float = 0
        self._last_hourly_summary: float = 0
        self._last_retention_cleanup: float = 0
        self._daily_start_equity: float = 0
        self._current_day: str = ""

    async def initialize(self) -> None:
        """Initialize all components and sync state."""
        setup_logging(self.settings.log_dir, self.settings.log_level)
        log.info("bot_initializing", mode=self.settings.bot_mode)

        # Database
        await self.db.initialize()

        # Exchange connections
        await self.market.initialize()

        # Validate symbols against exchange
        symbol_configs = get_enabled_symbols()
        candidate_symbols = [s.symbol for s in symbol_configs]
        self._symbols = self.market.get_available_symbols(candidate_symbols)

        if not self._symbols:
            log.error("no_valid_symbols", candidates=candidate_symbols)
            raise RuntimeError("No valid symbols found on exchange")

        log.info("symbols_validated", symbols=self._symbols)

        # Update risk manager with valid symbols
        self.risk_manager.allowed_symbols = self._symbols

        # Shared HTTP session for all data collectors (connection pooling)
        connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
        self._http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=10),
        )

        # Initialize new data collectors with shared session
        self.prediction_markets = PredictionMarketsCollector(session=self._http_session)
        self.top_traders = TopTraderCollector(session=self._http_session)
        self.news = NewsCollector(
            cryptopanic_api_key=getattr(self.settings, 'cryptopanic_api_key', ''),
            newsdata_api_key=getattr(self.settings, 'newsdata_api_key', ''),
            gnews_api_key=getattr(self.settings, 'gnews_api_key', ''),
            session=self._http_session,
        )

        # Aggregator
        self.aggregator = DataAggregator(
            market=self.market,
            websocket=self.websocket,
            onchain=self.onchain,
            sentiment=self.sentiment,
            macro=self.macro,
            prediction_markets=self.prediction_markets,
            top_traders=self.top_traders,
            news=self.news,
            symbols=self._symbols,
        )

        # Executor
        self.executor = TradeExecutor(
            exchange=self.market.trading_exchange,
            position_manager=self.position_manager,
            paper_mode=not self.settings.is_live,
            taker_fee=self.settings.exchange.taker_fee,
        )

        # Sync positions from exchange
        exchange_positions = await self.market.fetch_positions()
        self.position_manager.sync_from_exchange(exchange_positions)

        # Set starting equity
        balance = await self.market.fetch_balance()
        equity = balance.get("total", 0)
        self.position_manager.set_starting_equity(equity)
        self.memory.set_starting_capital(equity)
        self._daily_start_equity = equity

        # Load trade history into memory
        recent_trades = await self.db.get_recent_trades(20)
        for t in recent_trades:
            self.memory.record_trade(t)

        # Start WebSocket
        await self.websocket.start(self._symbols)

        # Telegram startup message
        await self.telegram.send_startup_message(
            mode=self.settings.bot_mode,
            equity=equity,
            symbols=self._symbols,
            positions=self.position_manager.position_count,
        )

        log.info("bot_initialized", equity=equity, symbols=self._symbols,
                 positions=self.position_manager.position_count)

    async def run(self) -> None:
        """Main trading loop."""
        await self.initialize()
        self._running = True

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: self._shutdown_event.set())

        log.info("bot_started")

        consecutive_tick_failures = 0
        max_tick_failures = 5
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    await self._tick()
                    consecutive_tick_failures = 0  # reset on success
                except Exception as e:
                    consecutive_tick_failures += 1
                    log.error(
                        "tick_error", error=str(e),
                        consecutive_failures=consecutive_tick_failures,
                        exc_info=True,
                    )
                    await self.telegram.send_error_alert(str(e))
                    # Silent-failure-hunter #11 — after repeated failures,
                    # transition to close-only mode (halt new entries) rather
                    # than trade on potentially corrupt state.
                    if consecutive_tick_failures >= max_tick_failures:
                        log.critical(
                            "tick_failure_threshold_exceeded",
                            failures=consecutive_tick_failures,
                        )
                        self.circuit_breakers._halt_trading(3600)  # 1h halt

                # Wait for next tick or shutdown
                wait_seconds = self._get_sleep_time()
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=min(wait_seconds, 30),  # Check at least every 30s
                    )
                    break  # Shutdown requested
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop

        finally:
            await self.shutdown()

    async def _halt_on_trigger_exit_failure(self, symbol: str, reason: str, result, detail: str) -> None:
        """Halt new entries and alert when an SL/TP close is not fully confirmed."""
        self.circuit_breakers._halt_trading(600)
        try:
            await self.telegram.send_circuit_breaker_alert(
                "protective_exit_close_failed",
                "HALT_NEW",
                (
                    f"{reason} close for {symbol} was not fully confirmed: {detail}. "
                    f"Order: {getattr(result, 'order_id', '') or 'unknown'}; "
                    f"status: {getattr(result, 'order_status', '') or 'unknown'}; "
                    f"filled: {getattr(result, 'filled_size', 0)}; "
                    f"remaining reported: {getattr(result, 'remaining_size', 0)}. "
                    "New entries halted for 10 minutes; reconcile exchange exposure."
                ),
            )
        except Exception as alert_error:
            log.error(
                "protective_exit_failure_alert_failed",
                symbol=symbol,
                reason=reason,
                error=str(alert_error),
            )

    async def _process_exit_triggers(self, ws_prices: dict[str, float]) -> None:
        """Close positions whose local SL/TP trigger has fired.

        The exchange executor is the only component allowed to close the
        position. Local state and trade history are updated only after the
        executor confirms a successful close.
        """
        sl_hits = self.position_manager.check_stop_losses(ws_prices)
        for symbol in sl_hits:
            price = ws_prices.get(symbol)
            if price is None or price <= 0:
                log.error("stop_loss_close_skipped_missing_price", symbol=symbol)
                continue

            result = await self.executor.close_position_for_trigger(
                symbol=symbol,
                current_price=price,
                reason="stop_loss",
            )
            if not result.success:
                log.error(
                    "stop_loss_exchange_close_failed",
                    symbol=symbol,
                    error=result.error,
                )
                await self._halt_on_trigger_exit_failure(
                    symbol, "stop_loss", result, result.error or "close not confirmed",
                )
                continue

            trade = result.trade_record
            if not trade:
                detail = "exchange close was reported successful but no trade record was returned"
                log.error(
                    "stop_loss_close_missing_trade_record",
                    symbol=symbol,
                    order_id=result.order_id,
                )
                await self._halt_on_trigger_exit_failure(
                    symbol, "stop_loss", result, detail,
                )
                continue

            exit_price = result.price or price
            self.memory.record_trade(trade)
            await self.db.save_trade(trade)
            self.circuit_breakers.record_loss(trade.get("pnl_pct", 0) / 100)

            # Phase 3 §3B.9 — record SL for cooldown.
            afml_cfg = getattr(self.settings, "afml", None)
            if (afml_cfg is not None
                    and getattr(afml_cfg, "enable_all", True)
                    and getattr(afml_cfg, "enable_sl_cooldown", True)):
                self.symbol_cooldown.on_stop_out(symbol, now=time.time())

            await self.telegram.notify_trade_close(
                symbol=symbol,
                side=trade["side"],
                entry_price=trade["entry_price"],
                exit_price=exit_price,
                pnl=trade["pnl"],
                pnl_pct=trade["pnl_pct"],
                duration=trade.get("duration", "N/A"),
                reasoning="Stop-loss hit",
            )

        tp_hits = self.position_manager.check_take_profits(ws_prices)
        for symbol in tp_hits:
            price = ws_prices.get(symbol)
            if price is None or price <= 0:
                log.error("take_profit_close_skipped_missing_price", symbol=symbol)
                continue

            result = await self.executor.close_position_for_trigger(
                symbol=symbol,
                current_price=price,
                reason="take_profit",
            )
            if not result.success:
                log.error(
                    "take_profit_exchange_close_failed",
                    symbol=symbol,
                    error=result.error,
                )
                await self._halt_on_trigger_exit_failure(
                    symbol, "take_profit", result, result.error or "close not confirmed",
                )
                continue

            trade = result.trade_record
            if not trade:
                detail = "exchange close was reported successful but no trade record was returned"
                log.error(
                    "take_profit_close_missing_trade_record",
                    symbol=symbol,
                    order_id=result.order_id,
                )
                await self._halt_on_trigger_exit_failure(
                    symbol, "take_profit", result, detail,
                )
                continue

            exit_price = result.price or price
            self.memory.record_trade(trade)
            await self.db.save_trade(trade)
            await self.telegram.notify_trade_close(
                symbol=symbol,
                side=trade["side"],
                entry_price=trade["entry_price"],
                exit_price=exit_price,
                pnl=trade["pnl"],
                pnl_pct=trade["pnl_pct"],
                duration=trade.get("duration", "N/A"),
                reasoning="Take-profit hit",
            )

    async def _process_margin_force_close(self, symbol: str, current_price: float) -> None:
        """Close a margin-unsafe position only after exchange fill confirmation."""
        result = await self.executor.close_position_for_trigger(
            symbol=symbol,
            current_price=current_price,
            reason="margin_buffer_force_close",
        )

        if not result.success or not result.trade_record:
            error = result.error or "Exchange close returned no completed trade record"
            uncertain = getattr(result, "state_uncertain", False)
            halt_seconds = 3600 if uncertain else 600
            remaining_note = (
                f"up to {result.remaining_size} may remain exposed; "
                "actual exchange exposure is unknown"
                if uncertain else f"remaining open: {result.remaining_size}"
            )
            log.critical(
                "margin_buffer_exchange_close_failed",
                symbol=symbol,
                order_id=result.order_id,
                order_status=result.order_status,
                filled_size=result.filled_size,
                remaining_size=result.remaining_size,
                state_uncertain=uncertain,
                error=error,
            )
            # Unknown or partial close leaves exposure unresolved; block new entries.
            self.circuit_breakers._halt_trading(halt_seconds)
            try:
                await self.telegram.send_circuit_breaker_alert(
                    "margin_force_close_failed",
                    "HALT_NEW",
                    (
                        f"Margin force-close for {symbol} was not fully confirmed: "
                        f"{error}. Order: {result.order_id or 'unknown'}; "
                        f"status: {result.order_status or 'unknown'}; "
                        f"confirmed filled: {result.filled_size}; "
                        f"{remaining_note}. New entries halted for {halt_seconds} seconds."
                    ),
                )
            except Exception as alert_error:
                log.error(
                    "margin_force_close_alert_failed",
                    symbol=symbol,
                    error=str(alert_error),
                )
            return

        trade = result.trade_record
        exit_price = result.price or current_price
        self.memory.record_trade(trade)
        await self.db.save_trade(trade)
        await self.telegram.notify_trade_close(
            symbol=symbol,
            side=trade["side"],
            entry_price=trade["entry_price"],
            exit_price=exit_price,
            pnl=trade["pnl"],
            pnl_pct=trade["pnl_pct"],
            duration=trade.get("duration", "N/A"),
            reasoning="Margin liquidation-buffer force-close",
        )

    async def _process_emergency_close_results(self, results, reason: str) -> None:
        """Persist and report only fully confirmed circuit-breaker closes."""
        for result in results:
            trade = result.trade_record
            if not result.success or not trade:
                error = result.error or (
                    "Executor reported a close without a completed trade record"
                    if result.success else
                    "Emergency close was not fully confirmed"
                )
                log.critical(
                    "circuit_breaker_emergency_close_incomplete",
                    symbol=result.symbol,
                    order_id=result.order_id,
                    order_status=result.order_status,
                    filled_size=result.filled_size,
                    remaining_size=result.remaining_size,
                    error=error,
                    breaker_reason=reason,
                )
                # Failed/partial close leaves exposure open; prevent new risk.
                self.circuit_breakers._halt_trading(3600)
                remaining_note = (
                    f"up to {result.remaining_size} may remain exposed; "
                    "actual exchange exposure is unknown"
                    if getattr(result, "state_uncertain", False)
                    else f"remaining open: {result.remaining_size}"
                )
                try:
                    await self.telegram.send_circuit_breaker_alert(
                        "emergency_close_incomplete",
                        "HALT_NEW",
                        (
                            f"Circuit-breaker close for {result.symbol} was not "
                            f"fully confirmed/accounted: {error}. "
                            f"Confirmed filled: {result.filled_size}; "
                            f"{remaining_note}. Trigger: {reason}"
                        ),
                    )
                except Exception as alert_error:
                    log.error(
                        "emergency_close_failure_alert_failed",
                        symbol=result.symbol,
                        error=str(alert_error),
                    )
                continue

            # Attempt each reporting destination independently. A failure in
            # one destination must not suppress the others or later results.
            reporting_errors: list[str] = []
            try:
                self.memory.record_trade(trade)
            except Exception as memory_error:
                reporting_errors.append(f"memory: {memory_error}")
                log.critical(
                    "emergency_close_memory_record_failed",
                    symbol=result.symbol,
                    order_id=result.order_id,
                    error=str(memory_error),
                )

            try:
                await self.db.save_trade(trade)
            except Exception as db_error:
                reporting_errors.append(f"database: {db_error}")
                log.critical(
                    "emergency_close_trade_persistence_failed",
                    symbol=result.symbol,
                    order_id=result.order_id,
                    error=str(db_error),
                )

            try:
                await self.telegram.notify_trade_close(
                    symbol=result.symbol,
                    side=trade.get("side", result.side),
                    entry_price=trade.get("entry_price", 0),
                    exit_price=result.price or trade.get("exit_price", 0),
                    pnl=trade.get("pnl", 0),
                    pnl_pct=trade.get("pnl_pct", 0),
                    duration=trade.get("duration", "N/A"),
                    reasoning=f"Circuit-breaker emergency close: {reason}",
                )
            except Exception as notify_error:
                reporting_errors.append(f"telegram: {notify_error}")
                log.error(
                    "emergency_close_notification_failed",
                    symbol=result.symbol,
                    order_id=result.order_id,
                    error=str(notify_error),
                )

            if reporting_errors:
                # Execution happened, but incomplete accounting/reporting
                # still requires a halt and an explicit operator alert.
                self.circuit_breakers._halt_trading(3600)
                try:
                    await self.telegram.send_circuit_breaker_alert(
                        "emergency_close_reporting_failed",
                        "HALT_NEW",
                        (
                            f"Exchange confirmed emergency close for {result.symbol}, "
                            f"but post-close reporting failed: "
                            f"{'; '.join(reporting_errors)}. Trigger: {reason}"
                        ),
                    )
                except Exception as alert_error:
                    log.error(
                        "emergency_close_reporting_alert_failed",
                        symbol=result.symbol,
                        error=str(alert_error),
                    )

    async def _tick(self) -> None:
        """Single iteration of the main loop."""
        now = time.time()

        # --- 1. Check day change ---
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_day:
            if self._current_day:
                await self._end_of_day()
            self._current_day = today
            balance = await self.market.fetch_balance()
            self._daily_start_equity = balance.get("total", 0)

        # --- 2. Update prices from WebSocket ---
        ws_prices = {}
        for sym in self._symbols:
            state = self.websocket.get_price_state(sym)
            if state and state.current_price > 0:
                ws_prices[sym] = state.current_price
        self.position_manager.update_prices(ws_prices)

        # --- 3. Check SL/TP hits through the exchange executor ---
        await self._process_exit_triggers(ws_prices)

        # --- 4. Circuit breaker checks ---
        balance = await self.market.fetch_balance()
        equity = balance.get("total", 0) + self.position_manager.total_unrealized_pnl
        self.position_manager.update_equity(equity)

        # Phase 3 §3B.12 — daily peak tracker
        afml_cfg = getattr(self.settings, "afml", None)
        if (afml_cfg is not None
                and getattr(afml_cfg, "enable_all", True)
                and getattr(afml_cfg, "enable_daily_peak_anchor", True)):
            self.daily_peak.update(equity=equity, now=now)

        # Feed data into MarketState (regime) and USDTMonitor — these
        # aggregators are no-ops until they actually see live signals.
        if afml_cfg is not None and getattr(afml_cfg, "enable_all", True):
            # Regime inputs: derive a compact vol_z from observed price change
            # and use funding-rate abs as a news-surge proxy. These are crude
            # proxies but meet the "something is better than nothing" bar.
            primary_sym = self._symbols[0] if self._symbols else None
            vol_z_proxy = 0.0
            if primary_sym:
                state = self.websocket.get_price_state(primary_sym)
                if state and getattr(state, "change_15m_pct", None) is not None:
                    # Normalize 15m % change by a conservative 1%-per-15m baseline
                    vol_z_proxy = abs(state.change_15m_pct) / 1.0
            news_surge_proxy = 0.0
            try:
                self.market_state.regime_update(
                    vol_z=float(vol_z_proxy),
                    news_surge=float(news_surge_proxy),
                    sentiment_z=0.0,
                    oi_delta=0.0,
                )
            except Exception as e:
                log.warning("regime_update_failed", error=str(e))

            # Feed USDT monitor with whatever stablecoin price we have. In
            # paper mode there's no dedicated oracle feed; we pass an empty
            # dict so the monitor flips to "stale" after threshold.
            try:
                self.usdt_monitor.record({}, now=now)
            except Exception as e:
                log.warning("usdt_monitor_update_failed", error=str(e))

        # Phase 3 §3B.2 — margin-ratio monitor; force-close if within buffer
        if (afml_cfg is not None
                and getattr(afml_cfg, "enable_all", True)
                and getattr(afml_cfg, "enable_margin_monitor", True)):
            positions_snapshot = self.position_manager.positions_for_prompt()
            try:
                margin_state = compute_margin_state(
                    positions=positions_snapshot, balance=balance,
                )
                self._latest_margin_state = margin_state
                force_buf = float(getattr(afml_cfg, "margin_force_close_buffer", 0.15))
                if (margin_state.positions
                        and margin_state.min_distance_to_liquidation_pct < force_buf):
                    # Emergency: close the most-at-risk position
                    worst = min(
                        margin_state.positions,
                        key=lambda p: p.distance_to_liquidation_pct,
                    )
                    log.critical(
                        "margin_buffer_force_close",
                        symbol=worst.symbol,
                        distance_pct=worst.distance_to_liquidation_pct,
                        buffer=force_buf,
                    )
                    price = ws_prices.get(worst.symbol, worst.mark_price)
                    await self._process_margin_force_close(worst.symbol, price)
            except Exception as e:
                # Silent-failure-hunter P0 #1: don't swallow — set state to None
                # (forces context_builder to omit ACCOUNT_MARGIN) and escalate.
                self._latest_margin_state = None
                log.critical("margin_monitor_error", error=str(e), exc_info=True)
                try:
                    await self.telegram.send_circuit_breaker_alert(
                        "margin_monitor_failure", "HALT_NEW",
                        f"Margin monitor crashed: {e}. Halting new entries until next successful compute.",
                    )
                except Exception:
                    pass  # Telegram failure must not block the loop
                # Halt new-entry circuit breaker for 10 minutes — margin unknown = unsafe
                self.circuit_breakers._halt_trading(600)

        # Get ATR for volatility check
        avg_atr_pct = 1.0  # default
        current_atr_pct = 1.0
        primary_sym = self._symbols[0] if self._symbols else None
        if primary_sym:
            state = self.websocket.get_price_state(primary_sym)
            # ATR from indicators would be more accurate, but this is a quick check

        # Get spread
        spread_pct = 0
        if primary_sym and primary_sym in (self.aggregator._symbols if self.aggregator else []):
            ob = await self.market.fetch_orderbook_summary(primary_sym, 5)
            spread_pct = ob.spread_pct / 100 if ob else 0

        # Get funding
        funding_rate = 0
        if primary_sym:
            fd = await self.market.fetch_funding_rate(primary_sym)
            funding_rate = fd.funding_rate if fd else 0

        daily_pnl_pct = (equity - self._daily_start_equity) / self._daily_start_equity if self._daily_start_equity else 0

        breaker_events = self.circuit_breakers.check_all(
            current_equity=equity,
            peak_equity=self.position_manager.peak_equity,
            daily_pnl_pct=daily_pnl_pct,
            current_atr_pct=current_atr_pct,
            avg_atr_pct=avg_atr_pct,
            spread_pct=spread_pct,
            funding_rate=funding_rate,
            ws_seconds_since_update=self.websocket.seconds_since_last_update,
            llm_consecutive_failures=self.llm_client.consecutive_failures,
        )

        for event in breaker_events:
            await self.db.save_circuit_breaker_event(
                event.breaker_type.value, event.action.value,
                event.reason, event.details,
            )
            await self.telegram.send_circuit_breaker_alert(
                event.breaker_type.value, event.action.value, event.reason,
            )
            if event.action == BreakerAction.CLOSE_ALL:
                results = await self.executor.emergency_close_all()
                await self._process_emergency_close_results(results, event.reason)
            elif event.action == BreakerAction.TIGHTEN_STOPS:
                for pos in self.position_manager.open_positions:
                    self.position_manager.update_stop_loss(pos.symbol, pos.entry_price)

        # --- 5. LLM analysis (if scheduler says go) ---
        price_change_15m = 0
        if primary_sym:
            state = self.websocket.get_price_state(primary_sym)
            price_change_15m = abs(state.change_15m_pct) if state else 0

        # Phase 1 §3.2 — CUSUM event filter fires on structural price breaks.
        # Independent of the 15m price-change trigger (an OR: either one wakes LLM).
        afml = getattr(self.settings, "afml", None)
        cusum_enabled = (
            afml is not None
            and getattr(afml, "enable_all", True)
            and getattr(afml, "enable_cusum", True)
        )
        if cusum_enabled:
            fired_syms: list[str] = []
            for sym, px in ws_prices.items():
                if sym not in self._cusum_filters:
                    self._cusum_filters[sym] = CUSUMFilter(
                        h=None,
                        k=float(getattr(afml, "cusum_k", 2.5)),
                        ewma_halflife=float(getattr(afml, "cusum_halflife", 20.0)),
                        min_periods=int(getattr(afml, "cusum_min_periods", 20)),
                    )
                # Only update on a fresh price (avoid repeated same-tick calls)
                if self._last_cusum_price.get(sym) != px:
                    self._last_cusum_price[sym] = px
                    event = self._cusum_filters[sym].update(px)
                    if event != 0:
                        fired_syms.append(sym)
            if fired_syms:
                self.scheduler.force_next_call()
                self._last_cusum_event_time = now
                log.info("cusum_event_fired", symbols=fired_syms)

        should_call = self.scheduler.should_call(
            has_open_position=self.position_manager.has_positions,
            price_change_15m_pct=price_change_15m,
            budget_ok=self.cost_tracker.can_afford_call(),
        )

        if should_call and not self.circuit_breakers.is_trading_halted:
            await self._run_llm_analysis(equity)

        # --- 6. Periodic tasks ---
        # Hourly summary
        if (self.settings.telegram.hourly_summary_enabled
                and self.position_manager.has_positions
                and now - self._last_hourly_summary > 3600):
            await self.telegram.send_hourly_summary(
                equity=equity,
                positions=self.position_manager.positions_for_prompt(),
                unrealized_pnl=self.position_manager.total_unrealized_pnl,
            )
            self._last_hourly_summary = now

        # Retention cleanup (once per day)
        if now - self._last_retention_cleanup > 86400:
            retention = RetentionManager(self.db.db)
            await retention.run_cleanup()
            self._last_retention_cleanup = now

    async def _halt_on_uncertain_execution(self, result) -> None:
        """Prevent further entries when an exchange order's outcome is unknown."""
        self.circuit_breakers._halt_trading(3600)
        try:
            await self.telegram.send_circuit_breaker_alert(
                "exchange_order_state_unknown",
                "HALT_NEW",
                (
                    f"Exchange order state is unknown for {result.symbol}. "
                    f"Order: {result.order_id or 'unknown'}; "
                    f"status: {result.order_status or 'unknown'}; "
                    f"confirmed filled: {result.filled_size}; "
                    f"up to {result.remaining_size} may remain exposed. "
                    f"Details: {result.error} New entries halted for one hour; "
                    "reconcile exchange orders and positions before resuming."
                ),
            )
        except Exception as alert_error:
            log.error(
                "unknown_order_state_alert_failed",
                symbol=result.symbol,
                order_id=result.order_id,
                error=str(alert_error),
            )

    async def _run_llm_analysis(self, equity: float) -> None:
        """Run a full LLM analysis cycle."""
        now = time.time()
        fetch_onchain = (now - self._last_onchain_fetch) > self.settings.onchain_interval
        fetch_macro = (now - self._last_macro_fetch) > self.settings.macro_interval
        fetch_pm = (now - self._last_prediction_markets_fetch) > 600   # every 10 min
        fetch_tt = (now - self._last_top_traders_fetch) > 1800          # every 30 min
        fetch_news = (now - self._last_news_fetch) > 300                # every 5 min

        # Collect ALL data concurrently
        snapshot = await self.aggregator.collect_snapshot(
            fetch_onchain=fetch_onchain,
            fetch_sentiment=True,
            fetch_macro=fetch_macro,
            fetch_prediction_markets=fetch_pm,
            fetch_top_traders=fetch_tt,
            fetch_news=fetch_news,
        )

        if fetch_onchain:
            self._last_onchain_fetch = now
        if fetch_macro:
            self._last_macro_fetch = now
        if fetch_pm:
            self._last_prediction_markets_fetch = now
        if fetch_tt:
            self._last_top_traders_fetch = now
        if fetch_news:
            self._last_news_fetch = now

        if not snapshot.has_critical_data:
            log.warning("insufficient_data_for_llm")
            return

        # Call LLM
        try:
            start = time.time()
            response = await self.llm_client.analyze(snapshot)
            elapsed_ms = int((time.time() - start) * 1000)

            # Save LLM call to database
            llm_call_id = await self.db.save_llm_call(
                prompt="(see context builder)",
                response_json=response.model_dump_json(),
                input_tokens=0,  # tracked by cost_tracker
                cached_tokens=0,
                output_tokens=0,
                cost_usd=0,
                execution_time_ms=elapsed_ms,
            )

            # Save decision
            primary_action = response.actions[0] if response.actions else None
            await self.db.save_decision(
                action=primary_action.action.value if primary_action else "hold",
                symbol=primary_action.symbol if primary_action else "N/A",
                trend=response.market_assessment.trend,
                confidence=response.market_assessment.confidence,
                reasoning=primary_action.reasoning if primary_action else response.risk_notes,
                llm_call_id=llm_call_id,
            )

            # Validate and execute actions
            current_prices = {
                sym: s.ticker.last
                for sym, s in snapshot.symbols.items()
                if s.ticker
            }

            # Phase 3 §3B.3 — slippage-aware sizing inputs
            orderbook_depths: dict[str, float] = {}
            spread_pcts: dict[str, float] = {}
            for sym, s in snapshot.symbols.items():
                if s.orderbook is not None:
                    depth = (s.orderbook.bid_depth or 0) + (s.orderbook.ask_depth or 0)
                    orderbook_depths[sym] = float(depth)
                    # s.orderbook.spread_pct is percent (e.g., 0.02 means 0.02%); normalize to fraction
                    spread_frac = float(s.orderbook.spread_pct or 0) / 100.0
                    spread_pcts[sym] = spread_frac

            validations = self.risk_manager.validate_response(
                response=response,
                equity=equity,
                open_positions=self.position_manager.positions_for_prompt(),
                current_prices=current_prices,
                orderbook_depths=orderbook_depths,
                spread_pcts=spread_pcts,
            )

            llm_confidence = response.market_assessment.confidence
            afml_cfg = getattr(self.settings, "afml", None)
            afml_on = (
                afml_cfg is not None
                and getattr(afml_cfg, "enable_all", True)
                and getattr(afml_cfg, "enable_triple_barrier", True)
            )
            max_holding = float(getattr(afml_cfg, "max_holding_s", 259_200)) if afml_on else None

            for action, validation in zip(response.actions, validations):
                if action.action == ActionType.HOLD:
                    continue

                result = await self.executor.execute(
                    action=action,
                    validation=validation,
                    current_price=current_prices.get(action.symbol, 0),
                    equity=equity,
                    confidence=llm_confidence,
                    max_holding_s=max_holding,
                )

                if result.state_uncertain:
                    await self._halt_on_uncertain_execution(result)
                    # Do not submit later actions from this LLM response while
                    # an earlier exchange order may still be live.
                    break

                if result.success and action.action in (
                    ActionType.OPEN_LONG, ActionType.OPEN_SHORT
                ):
                    await self.telegram.notify_trade_open(
                        symbol=action.symbol,
                        side=result.side,
                        size=result.size * result.price,
                        leverage=action.leverage or 2,
                        entry_price=result.price,
                        stop_loss=action.stop_loss,
                        take_profit=action.take_profit,
                        reasoning=action.reasoning,
                    )
                elif result.success and action.action == ActionType.CLOSE:
                    # Trade already recorded by the executor's close path.
                    pass

        except LLMBudgetExceededError:
            log.warning("llm_budget_exceeded")
        except Exception as e:
            log.error("llm_analysis_error", error=str(e), exc_info=True)

    async def _end_of_day(self) -> None:
        """End-of-day tasks: save summary, send report."""
        balance = await self.market.fetch_balance()
        equity = balance.get("total", 0) + self.position_manager.total_unrealized_pnl
        pnl = equity - self._daily_start_equity
        pnl_pct = pnl / self._daily_start_equity if self._daily_start_equity else 0

        stats = await self.db.get_trade_stats()
        cost = self.cost_tracker.get_summary()

        summary = {
            "date": self._current_day,
            "starting_equity": self._daily_start_equity,
            "ending_equity": equity,
            "pnl": round(pnl, 4),
            "pnl_pct": round(pnl_pct, 4),
            "trades_count": stats["total_trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "llm_calls": cost["calls_today"],
            "llm_cost": cost["today_spend"],
            "max_drawdown_pct": 0,  # TODO: track intraday
        }
        await self.db.save_daily_summary(summary)

        if self.settings.telegram.daily_summary_enabled:
            await self.telegram.send_daily_summary(
                date=self._current_day,
                starting_equity=self._daily_start_equity,
                ending_equity=equity,
                pnl=pnl,
                pnl_pct=pnl_pct * 100,
                trades=stats["total_trades"],
                wins=stats["wins"],
                losses=stats["losses"],
                max_drawdown_pct=0,
                llm_cost=cost["today_spend"],
            )

    def _get_sleep_time(self) -> float:
        """Calculate how long to sleep until next action."""
        return self.scheduler.get_next_call_in(
            has_open_position=self.position_manager.has_positions,
            budget_ok=self.cost_tracker.can_afford_call(),
        )

    async def shutdown(self, reason: str = "manual") -> None:
        """Graceful shutdown."""
        log.info("bot_shutting_down", reason=reason)
        self._running = False

        await self.telegram.send_shutdown_message(reason)

        # Close components
        await self.websocket.stop()
        await self.market.close()
        await self.onchain.close()
        await self.sentiment.close()
        self.macro.close()
        if self.prediction_markets:
            await self.prediction_markets.close()
        if self.top_traders:
            await self.top_traders.close()
        if self.news:
            await self.news.close()
        if self._http_session:
            await self._http_session.close()
        await self.db.close()

        log.info("bot_shutdown_complete")
