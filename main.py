"""Mirasim traffic-based availability monitor: status on demand, change pushes."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from pathlib import Path

import httpx

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star

DEFAULT_API_URL = "https://mirasim.ai/api/status"
REQUEST_TIMEOUT = 20
# mirasim.ai refreshes its aggregation once a minute, a faster poll finds nothing.
MIN_POLL_INTERVAL = 60
DEFAULT_POLL_INTERVAL = 60
# One change of mind is enough at the official minute cadence; two keeps blips quiet.
DEFAULT_CONFIRM_POLLS = 2
# Availability thresholds (%). These are intentionally much looser than the
# site's 99/95: at 95% a service is already flaky for end users, so we let it
# count as normal until 90% and only call <50% a real outage.
THRESHOLD_GOOD = 90
THRESHOLD_WARN = 50
# Subscription entry meaning "every cohort of every model", including later models.
ALL_MODELS = "*"
ALL_WORDS = ("all", "*", "全部")
COHORT_ORDER = ("free", "paid", "cloud")
COHORT_NAMES = {"free": "免费", "paid": "付费", "cloud": "云端"}
TEMPLATE_PATH = Path(__file__).parent / "templates" / "status.html"
# PNG keeps the small text crisp; "ultra" is a 1.8x device pixel ratio on the
# official T2I service, which phones need once they scale the card down.
RENDER_OPTIONS = {
    "type": "png",
    "full_page": True,
    "device_scale_factor_level": "ultra",
}

STATUS_ICONS = {"ok": "✅", "warn": "⚠️", "down": "❌", "nodata": "❔"}
STATUS_NAMES = {"ok": "正常", "warn": "不稳定", "down": "中断", "nodata": "无数据"}
# Problems sort first so the top of the list is all a reader needs to check.
STATUS_ORDER = {"down": 0, "warn": 1, "nodata": 2, "unknown": 2, "ok": 3}
CHANGE_ICONS = {"ok": "🟢", "warn": "🟡", "down": "🔴", "nodata": "⚪"}
# Failure classes reported at the agent level, worded the way the site does.
REASON_NAMES = {
    "throttle": "限流",
    "outage": "故障",
    "capacity": "容量不足",
    "error": "错误",
}

USAGE = (
    "Mirasim 模型状态\n"
    "/mirasim — 查看模型状态\n"
    "/mirasim all — 查看全部模型\n"
    "/mirasim <模型> — 查看单个模型各池详情\n"
    "/mirasim <模型>@<free|paid|cloud> — 只看指定池\n"
    "/mirasim sub <模型>[@池]|all — 本会话订阅状态变化推送\n"
    "/mirasim unsub <模型>[@池]|all — 取消订阅\n"
    "/mirasim list — 查看本会话的订阅\n"
    "模型名可只写能唯一确定的一部分，如 opus-5-5。"
)


def _parse_time(value: object) -> datetime | None:
    """Parse an ISO 8601 timestamp from the payload.

    Args:
        value: Raw field value, normally like ``2026-10-06T07:39:03.125Z``.

    Returns:
        An aware datetime, or None when the value is missing or malformed.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        # datetime.fromisoformat() only accepts a trailing "Z" from 3.11 on.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_time(value: object) -> str:
    """Format a timestamp in local time, dropping the date for today.

    Args:
        value: Raw ISO 8601 timestamp.

    Returns:
        Text such as ``20:26`` or ``10-02 20:26``; ``—`` when unknown.
    """
    moment = _parse_time(value)
    if moment is None:
        return "—"
    local = moment.astimezone()
    today = datetime.now().astimezone().date()
    return local.strftime("%H:%M" if local.date() == today else "%m-%d %H:%M")


def _fmt_duration(seconds: float) -> str:
    """Format a duration with its two most significant units.

    Args:
        seconds: Length in seconds; negative values count as zero.

    Returns:
        Text such as ``2小时21分钟``.
    """
    days, rest = divmod(int(max(seconds, 0)), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}天{hours}小时"
    if hours:
        return f"{hours}小时{minutes}分钟"
    if minutes:
        return f"{minutes}分钟"
    return f"{secs}秒"


def _fmt_seconds(value: object) -> str:
    """Format a seconds count, e.g. ``5.2s``; ``—`` when unknown.

    Args:
        value: Seconds as a number; the site reports ``null`` when it has no data.

    Returns:
        The formatted latency, or ``—``.
    """
    if not isinstance(value, (int, float)) or value < 0:
        return "—"
    return f"{value:.1f}s"


def _fmt_rate(value: object, *, short: bool = False) -> str:
    """Format an availability percentage; ``—`` when unknown.

    Args:
        value: Percentage as a number.
        short: Round like the site does, ``100%`` instead of ``100.0%``.

    Returns:
        The formatted percentage.
    """
    if not isinstance(value, (int, float)):
        return "—"
    return f"{round(value, 1):g}%" if short else f"{value:.1f}%"


def _cell_status(value: object, good: float, warn: float) -> str:
    """Translate one 30-minute cell to a colour bucket.

    Cells are stored as availability ×10 (1000 = 100%), and -1 marks a gap.

    Args:
        value: Raw cell value.
        good: Availability percentage still counted as 正常.
        warn: Availability percentage still counted as 不稳定.

    Returns:
        One of ``ok``, ``warn``, ``down``, ``unknown``.
    """
    if not isinstance(value, (int, float)) or value < 0:
        return "unknown"
    return "ok" if value >= 10 * good else ("warn" if value >= 10 * warn else "down")


def _worst_status(statuses: list[str]) -> str:
    """Pick the worst of several status labels.

    Args:
        statuses: Labels to compare.

    Returns:
        The worst one under STATUS_ORDER.
    """
    return min(statuses, key=lambda s: STATUS_ORDER.get(s, 99), default="unknown")


def _history_bars(cells: object, data: dict) -> list[str]:
    """Build the per-cell status list used by stripes and outage bounds.

    The plugin ignores the site's own thresholds (99/95) and applies the
    looser THRESHOLD_GOOD/WARN — see their comments for why.

    Args:
        cells: Raw ``cells`` array of a model.
        data: Whole payload; reserved for future per-payload overrides.

    Returns:
        One ``ok``/``warn``/``down``/``unknown`` per cell, oldest first.
    """
    if not isinstance(cells, list):
        return []
    return [_cell_status(value, THRESHOLD_GOOD, THRESHOLD_WARN) for value in cells]


def _cell_time(data: dict, index: int) -> datetime | None:
    """Convert a cell index to its starting timestamp.

    Args:
        data: Whole payload, for ``cellsStart``/``cellSeconds``.
        index: Position inside ``cells``, oldest first.

    Returns:
        An aware datetime, or None when the payload lacks the timing fields.
    """
    start = _parse_time(data.get("cellsStart"))
    cell_seconds = data.get("cellSeconds")
    if start is None or not isinstance(cell_seconds, (int, float)):
        return None
    return datetime.fromtimestamp(
        start.timestamp() + index * cell_seconds, tz=start.tzinfo
    )


def _last_healthy_time(bars: list[str], data: dict) -> datetime | None:
    """Find the most recent non-down cell timestamp, or None.

    Args:
        bars: Per-cell statuses ordered oldest first.
        data: Whole payload.

    Returns:
        An aware datetime, or None when nothing healthy is visible.
    """
    for i in range(len(bars) - 1, -1, -1):
        if bars[i] in ("ok", "warn"):
            return _cell_time(data, i)
    return None


def _resolve_id(ids: list[str], query: str, hint: str) -> str:
    """Resolve user input to exactly one model ID.

    A case-insensitive exact match wins; otherwise the input must be part of
    exactly one ID, so ``opus-5-5`` finds ``claude-opus-5-5``.

    Args:
        ids: Candidate model IDs.
        query: What the user typed.
        hint: Appended to the "not found" message to point at the next step.

    Returns:
        The matched model ID.

    Raises:
        LookupError: Nothing or several IDs match; the message is user-facing.
    """
    needle = query.lower()
    for model_id in ids:
        if model_id.lower() == needle:
            return model_id
    hits = sorted(model_id for model_id in ids if needle in model_id.lower())
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise LookupError(f"没有找到匹配「{query}」的模型，{hint}")
    raise LookupError(
        f"「{query}」匹配到多个模型：{'、'.join(hits)}，请输入更完整的 ID。"
    )


def _split_target(text: str) -> tuple[str, str | None]:
    """Split ``model@cohort`` into its two parts.

    Args:
        text: Raw target from the user.

    Returns:
        ``(model_part, cohort_part or None)``.
    """
    if "@" not in text:
        return text.strip(), None
    model_part, cohort_part = text.rsplit("@", 1)
    return model_part.strip(), cohort_part.strip().lower() or None


def _covers(entry: str, key: str) -> bool:
    """Tell whether a subscription entry matches a ``model@cohort`` change key.

    Args:
        entry: Stored subscription, ``model``, ``model@cohort`` or ``*``.
        key: Change key, always ``model@cohort``.

    Returns:
        True when the entry wants this key.
    """
    if entry == ALL_MODELS:
        return True
    model_part, cohort_part = _split_target(entry)
    key_model, key_cohort = _split_target(key)
    if key_model != model_part:
        return False
    return cohort_part is None or cohort_part == key_cohort


def _format_reasons(reasons: object) -> str:
    """Render the agent-level failure breakdown as a short parenthetical.

    Args:
        reasons: ``reasons`` list from the agent summary.

    Returns:
        Text such as ``(限流79%·故障21%)`` or empty string.
    """
    if not isinstance(reasons, list):
        return ""
    parts = []
    for item in reasons:
        if not isinstance(item, dict):
            continue
        cls = item.get("class")
        share = item.get("share")
        if not cls or not isinstance(share, (int, float)) or share < 1:
            continue
        label = REASON_NAMES.get(cls, str(cls))
        parts.append(f"{label}{round(share):g}%")
    return f"({'·'.join(parts)})" if parts else ""


def _now_status_of(model: dict) -> tuple[str, float | None]:
    """Read a model's current status and availability.

    The plugin recomputes the bucket from ``availability`` instead of trusting
    ``now.status``, because the site uses much stricter thresholds (99/95)
    than the plugin does (see THRESHOLD_GOOD/WARN).

    Args:
        model: Model dict from the payload.

    Returns:
        ``(status, availability)`` where status is ok/warn/down/nodata.
    """
    now = model.get("now") or {}
    availability = now.get("availability")
    if not isinstance(availability, (int, float)):
        return "nodata", None
    if availability >= THRESHOLD_GOOD:
        return "ok", availability
    if availability >= THRESHOLD_WARN:
        return "warn", availability
    return "down", availability


class MirasimStatus(Star):
    """Show Mirasim per-cohort model availability and push confirmed changes."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        # Session (unified_msg_origin) -> subscription entries (model[@cohort] or *).
        self._subs: dict[str, list[str]] = {}
        # model@cohort -> confirmed status as of the last poll, the push baseline.
        self._states: dict[str, str] = {}
        self._task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Restore persisted state and start the background poller."""
        self._subs = await self.get_kv_data("subscriptions", {}) or {}
        self._states = await self.get_kv_data("states", {}) or {}
        self._task = asyncio.create_task(self._poll_loop())

    async def terminate(self) -> None:
        """Stop the background poller."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _fetch_status(self) -> dict:
        """Fetch the latest snapshot from the monitor.

        Returns:
            The decoded status payload.

        Raises:
            httpx.HTTPError: The request failed or returned an error status.
            ValueError: The body is not a status document.
        """
        url = str(self.config.get("api_url") or DEFAULT_API_URL).strip()
        # httpx honours the proxy environment AstrBot sets from its settings.
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("cohorts"), list):
            raise ValueError("unexpected status payload")
        return data

    def _all_models(self, data: dict) -> dict[str, list[tuple[str, dict, dict]]]:
        """Group every model by ID with its per-cohort entries.

        Args:
            data: Status payload.

        Returns:
            ``{model_id: [(cohort_id, model_dict, agent_dict), ...]}``
            ordered by COHORT_ORDER.
        """
        result: dict[str, list[tuple[str, dict, dict]]] = {}
        cohort_index = {cid: i for i, cid in enumerate(COHORT_ORDER)}
        for cohort in data.get("cohorts") or []:
            cohort_id = cohort.get("id")
            if not cohort_id:
                continue
            for agent in cohort.get("agents") or []:
                for model in agent.get("models") or []:
                    model_id = model.get("id")
                    if not model_id:
                        continue
                    result.setdefault(model_id, []).append((cohort_id, model, agent))
        for entries in result.values():
            entries.sort(key=lambda e: cohort_index.get(e[0], len(COHORT_ORDER)))
        return result

    def _agent_by_id(self, data: dict, cohort_id: str, agent_id: str) -> dict | None:
        """Locate one agent blob in the payload.

        Args:
            data: Status payload.
            cohort_id: Cohort the agent belongs to.
            agent_id: Agent ID to look up.

        Returns:
            The agent dict, or None when absent.
        """
        for cohort in data.get("cohorts") or []:
            if cohort.get("id") != cohort_id:
                continue
            for agent in cohort.get("agents") or []:
                if agent.get("id") == agent_id:
                    return agent
        return None

    async def _poll_loop(self) -> None:
        """Poll the monitor forever and push confirmed status changes."""
        prev_data: dict | None = None
        while True:
            # Sleep first: right after startup the platform adapters may still
            # be connecting, and a push sent then would be lost.
            interval = max(
                MIN_POLL_INTERVAL,
                int(
                    self.config.get("poll_interval", DEFAULT_POLL_INTERVAL)
                    or DEFAULT_POLL_INTERVAL
                ),
            )
            await asyncio.sleep(interval)
            try:
                data = await self._fetch_status()
                await self._check_changes(data, prev_data)
                prev_data = data
            except Exception as exc:
                logger.warning(f"[Mirasim] Status poll failed: {exc!r}")

    async def _check_changes(self, data: dict, prev_data: dict | None = None) -> None:
        """Detect confirmed transitions and push them to subscribers.

        A transition is confirmed when the model's ``now.status`` has matched
        ``confirm_polls`` polls in a row *and* differs from the baseline. The
        first poll after ``initialize`` records the baseline silently so a
        restart doesn't fire a flood of stale changes.

        Args:
            data: Latest status payload.
            prev_data: Payload from the previous poll, or None.
        """
        confirm = max(
            1,
            int(
                self.config.get("confirm_samples", DEFAULT_CONFIRM_POLLS)
                or DEFAULT_CONFIRM_POLLS
            ),
        )

        grouped_now = self._all_models(data)
        grouped_prev = self._all_models(prev_data) if prev_data else {}

        # First poll ever: just record the baseline, no pushes.
        if not self._states:
            self._states = {
                f"{model_id}@{cohort_id}": _now_status_of(model)[0]
                for model_id, entries in grouped_now.items()
                for cohort_id, model, _a in entries
            }
            await self.put_kv_data("states", self._states)
            return

        changes: list[tuple[str, str, str, dict, dict, dict]] = []
        dirty = False
        for model_id, entries in grouped_now.items():
            for cohort_id, model, agent in entries:
                key = f"{model_id}@{cohort_id}"
                current, _availability = _now_status_of(model)
                previous_confirmed = self._states.get(key, current)

                # With confirm >= 2, the previous poll's now.status must also
                # have equalled current; otherwise the change hasn't settled.
                if confirm >= 2:
                    prev_entries = grouped_prev.get(model_id, [])
                    prev_map = {c: m for c, m, _a in prev_entries}
                    prev_model = prev_map.get(cohort_id)
                    if prev_model is None:
                        previous_now = current  # model just appeared
                    else:
                        previous_now, _ = _now_status_of(prev_model)
                    if previous_now != current:
                        continue  # still flickering

                if previous_confirmed == current:
                    continue
                # Confirmed transition.
                changes.append(
                    (key, previous_confirmed, current, model, cohort_id, agent)
                )
                self._states[key] = current
                dirty = True

        if not changes:
            return
        if dirty:
            await self.put_kv_data("states", self._states)
        logger.info(f"[Mirasim] Status changes: {[k for k, *_ in changes]}")

        cells_start = _parse_time(data.get("cellsStart"))
        cell_seconds = data.get("cellSeconds")

        lines_by_key: list[tuple[str, str]] = []
        for key, _old, new_status, model, cohort_id, agent in changes:
            model_id = key.split("@", 1)[0]
            icon = CHANGE_ICONS.get(new_status, "⚪")
            label = COHORT_NAMES.get(cohort_id, cohort_id)
            reason = _format_reasons(agent.get("reasons"))
            latency = model.get("latency") or {}
            _s, availability = _now_status_of(model)
            availability_text = _fmt_rate(availability, short=True)

            # Failure run inside the 24h cell window.
            # - For "down" we want where the current down streak started.
            # - For "ok" (recovery) we want the down streak that just ended.
            bars = _history_bars(model.get("cells"), data)
            outage_start: int | None = None
            began_before = False
            if bars:
                target = new_status if new_status == "down" else "down"
                i = len(bars) - 1
                # Skip trailing cells that don't match the target streak.
                while i >= 0 and bars[i] != target:
                    i -= 1
                # Walk back through the streak.
                while i >= 0 and bars[i] == target:
                    i -= 1
                # i is now the last index NOT in the streak; streak starts at i+1.
                if i + 1 < len(bars):
                    outage_start = i + 1
                    began_before = outage_start == 0 and bars[0] == target

            if new_status == "down":
                line = (
                    f"{icon} {model_id}（{label}）中断 {reason}"
                    f"（可用率 {availability_text}"
                )
                if (
                    outage_start is not None
                    and cells_start is not None
                    and isinstance(cell_seconds, (int, float))
                ):
                    since = datetime.fromtimestamp(
                        cells_start.timestamp() + outage_start * cell_seconds,
                        tz=cells_start.tzinfo,
                    )
                    line += f"，{_fmt_time(since.isoformat())} 起"
                line += "）"
            elif new_status == "warn":
                line = (
                    f"{icon} {model_id}（{label}）不稳定 {reason}"
                    f"（可用率 {availability_text}）"
                )
            elif new_status == "ok":
                p50 = _fmt_seconds(latency.get("p50"))
                line = (
                    f"{icon} {model_id}（{label}）已恢复，"
                    f"p50 {p50}，可用率 {availability_text}"
                )
                if outage_start is not None and isinstance(cell_seconds, (int, float)):
                    # Cells between outage_start and the newest ok cell are the
                    # failure run that just ended.
                    outage_secs = (len(bars) - outage_start - 1) * cell_seconds
                    if outage_secs > 0:
                        outage = _fmt_duration(outage_secs)
                        prefix = "超过" if began_before else "约"
                        line += f"，中断持续{prefix} {outage}"
            else:
                line = f"{icon} {model_id}（{label}）{STATUS_NAMES.get(new_status, new_status)}"
            lines_by_key.append((key, line))

        # Snapshot the sessions: a command may edit them while we await sends.
        for umo, wanted in list(self._subs.items()):
            lines = [
                line
                for key, line in lines_by_key
                if any(_covers(e, key) for e in wanted)
            ]
            if not lines:
                continue
            text = "Mirasim 状态变化\n" + "\n".join(lines)
            try:
                if not await self.context.send_message(
                    umo, MessageChain().message(text)
                ):
                    logger.warning(f"[Mirasim] No platform found to push to {umo}")
            except Exception as exc:
                logger.warning(f"[Mirasim] Push to {umo} failed: {exc!r}")

    @filter.command("mirasim")
    async def cmd_mirasim(
        self,
        event: AstrMessageEvent,
        action: str = "",
        target: str = "",
    ):
        """查看 Mirasim 模型状态，或管理本会话的状态推送订阅。"""
        act = action.strip().lower()
        target = target.strip()
        umo = event.unified_msg_origin

        if act in ("help", "帮助"):
            text = USAGE
        elif act in ("sub", "订阅"):
            text = await self._subscribe(umo, target)
        elif act in ("unsub", "取消订阅", "退订"):
            text = await self._unsubscribe(umo, target)
        elif act in ("list", "订阅列表"):
            wanted = self._subs.get(umo, [])
            if not wanted:
                text = "本会话还没有订阅任何模型，发送 /mirasim sub <模型> 订阅。"
            elif ALL_MODELS in wanted:
                text = "本会话已订阅全部模型（包括之后新增的模型）。"
            else:
                text = f"本会话订阅了 {len(wanted)} 项：\n" + "\n".join(
                    f"· {entry}" for entry in wanted
                )
        else:
            try:
                data = await self._fetch_status()
            except Exception as exc:
                logger.warning(f"[Mirasim] Status query failed: {exc!r}")
                text = f"获取 Mirasim 状态失败：{str(exc) or type(exc).__name__}"
            else:
                if act and act not in ALL_WORDS:
                    text = self._render_model(data, action.strip())
                else:
                    show_all = bool(act)
                    if self.config.get("render_image", True):
                        path = await self._render_overview_image(data, show_all)
                        if path:
                            yield event.image_result(path)
                            return
                    text = self._render_overview(data, show_all)
        yield event.plain_result(text)

    def _render_overview(self, data: dict, show_all: bool) -> str:
        """Render the status list of the configured models, or of all of them.

        Args:
            data: Status payload.
            show_all: Ignore the ``display_models`` setting.

        Returns:
            The reply text.
        """
        rows, missing, filtered = self._overview_rows(data, show_all)
        up = sum(
            1
            for _id, entries, _a in rows
            for status, _entry in entries
            if status == "ok"
        )
        total = sum(len(entries) for _id, entries, _a in rows)
        lines = [f"Mirasim 模型状态 · 可用 {up}/{total}"]
        if not rows:
            lines.append("暂无模型数据。")
        for model_id, entries, agent_name in rows:
            agent_suffix = f" — {agent_name}" if agent_name else ""
            lines.append(f"· {model_id}{agent_suffix}")
            for status, (cohort_id, model, _agent) in entries:
                availability = (model.get("now") or {}).get("availability")
                p50 = (model.get("latency") or {}).get("p50")
                detail = (
                    f"{_fmt_rate(availability, short=True)} · p50 {_fmt_seconds(p50)}"
                )
                lines.append(
                    f"  {STATUS_ICONS.get(status, '❔')} "
                    f"{COHORT_NAMES.get(cohort_id, cohort_id)} · {detail}"
                )
        if missing:
            lines.append(f"未找到：{'、'.join(missing)}")
        footer = f"检测于 {_fmt_time(data.get('generatedAt'))}"
        cell_seconds = data.get("cellSeconds")
        if isinstance(cell_seconds, (int, float)) and cell_seconds > 0:
            footer += f" · 每 {_fmt_duration(cell_seconds)}一格"
        lines.append(footer)
        if filtered:
            lines.append("仅展示配置的模型，发送 /mirasim all 查看全部。")
        return "\n".join(lines)

    def _overview_rows(
        self, data: dict, show_all: bool
    ) -> tuple[
        list[tuple[str, list[tuple[str, tuple[str, dict, dict]]], str]], list[str], bool
    ]:
        """Pick, filter and sort the models shown in the overview.

        Args:
            data: Status payload.
            show_all: Ignore the ``display_models`` setting.

        Returns:
            ``(model_id, entries, agent_name)`` rows with problems first. Each
            entry in ``entries`` is ``(status, (cohort_id, model, agent))``.
        """
        all_models = self._all_models(data)
        configured = [
            str(item).strip()
            for item in self.config.get("display_models") or []
            if str(item).strip()
        ]
        filtered = bool(configured) and not show_all
        missing: list[str] = []
        chosen: list[str] = []
        if filtered:
            lowered = {k.lower(): k for k in all_models}
            for model_id in dict.fromkeys(configured):
                if model_id.lower() in lowered:
                    chosen.append(lowered[model_id.lower()])
                else:
                    missing.append(model_id)
        else:
            chosen = list(all_models)

        rows: list[tuple[str, list[tuple[str, tuple[str, dict, dict]]], str, str]] = []
        for model_id in chosen:
            entries = all_models[model_id]
            tagged = [
                (_now_status_of(model)[0], (cohort_id, model, agent))
                for cohort_id, model, agent in entries
            ]
            worst = _worst_status([status for status, _e in tagged])
            agent_name = entries[0][2].get("name") or ""
            rows.append((model_id, tagged, agent_name, worst))
        rows.sort(key=lambda r: (STATUS_ORDER.get(r[3], 99), r[0]))
        return [(m, e, a) for m, e, a, _w in rows], missing, filtered

    async def _render_overview_image(self, data: dict, show_all: bool) -> str | None:
        """Render the overview as a card image through AstrBot's T2I service.

        Args:
            data: Status payload.
            show_all: Ignore the ``display_models`` setting.

        Returns:
            Path of the rendered image, or None when rendering failed and the
            caller should fall back to text.
        """
        rows, missing, filtered = self._overview_rows(data, show_all)
        counts: dict[str, int] = {}
        groups = []
        total_up = 0
        total_rows = 0
        for model_id, entries, agent_name in rows:
            cards = []
            for status, (cohort_id, model, agent) in entries:
                counts[status] = counts.get(status, 0) + 1
                total_rows += 1
                if status == "ok":
                    total_up += 1
                availability = (model.get("now") or {}).get("availability")
                p50 = (model.get("latency") or {}).get("p50")
                metrics = (
                    f"{_fmt_rate(availability, short=True)} · p50 {_fmt_seconds(p50)}"
                )
                if status == "down":
                    note = _format_reasons(agent.get("reasons")) or "服务中断"
                elif status == "warn":
                    note = "可用率波动"
                elif status == "nodata":
                    note = "暂未采集到流量"
                else:
                    note = "运行正常"
                bars = _history_bars(model.get("cells"), data)
                cards.append(
                    {
                        "cohort": cohort_id,
                        "cohort_label": COHORT_NAMES.get(cohort_id, cohort_id),
                        "status": status if status in STATUS_ORDER else "unknown",
                        "label": STATUS_NAMES.get(status, status),
                        "metrics": metrics,
                        "note": note,
                        "history": bars,
                    }
                )
            groups.append(
                {
                    "id": model_id,
                    "agent": agent_name,
                    "worst": _worst_status([c["status"] for c in cards]),
                    "cards": cards,
                }
            )
        interval = data.get("cellSeconds")
        tmpl_data = {
            "groups": groups,
            "counts": counts,
            "total_up": total_up,
            "total_rows": total_rows,
            "missing": missing,
            "filtered": filtered,
            "checked_at": _fmt_time(data.get("generatedAt")),
            "interval": (
                _fmt_duration(interval)
                if isinstance(interval, (int, float)) and interval > 0
                else ""
            ),
            "source": "mirasim.ai",
        }

        try:
            path = await self.html_render(
                TEMPLATE_PATH.read_text(encoding="utf-8"),
                tmpl_data,
                return_url=False,
                options=RENDER_OPTIONS,
            )
            with open(path, "rb") as file:
                head = file.read(8)
        except Exception as exc:
            logger.warning(f"[Mirasim] Image render failed: {exc!r}")
            return None
        # The T2I client saves any response body, error pages included.
        if not head.startswith((b"\x89PNG", b"\xff\xd8")):
            logger.warning(f"[Mirasim] T2I returned a non-image body: {head!r}")
            return None
        return path

    def _render_model(self, data: dict, query: str) -> str:
        """Render the detail view of one model, or one of its cohorts.

        Args:
            data: Status payload.
            query: The user's ``model[@cohort]`` input.

        Returns:
            The reply text.
        """
        model_id, entries, error = self._resolve_model_and_entry(data, query)
        if error:
            return error
        _model_part, cohort_part = _split_target(query)
        title = model_id + (f"@{cohort_part}" if cohort_part else "")
        lines = [f"· {title}"]

        for cohort_id, model, agent in entries:
            label = COHORT_NAMES.get(cohort_id, cohort_id)
            status, availability = _now_status_of(model)
            icon = STATUS_ICONS.get(status, "❔")
            name = STATUS_NAMES.get(status, status)
            reason = _format_reasons(agent.get("reasons"))
            latency = model.get("latency") or {}
            availability_stats = model.get("availability") or {}
            same_model = model.get("sameModel")
            intel = model.get("intel")

            header = f"{icon} {label}池 {name}"
            if status in ("down", "warn") and isinstance(availability, (int, float)):
                header += f"（可用率 {_fmt_rate(availability, short=True)}）"
            if reason:
                header += f" {reason}"
            lines.append(header)

            detail_bits = []
            if isinstance(latency.get("p50"), (int, float)):
                detail_bits.append(f"p50 {_fmt_seconds(latency['p50'])}")
            if isinstance(latency.get("p95"), (int, float)):
                detail_bits.append(f"p95 {_fmt_seconds(latency['p95'])}")
            if detail_bits:
                lines.append(f"  延迟 {' · '.join(detail_bits)}")
            h24 = availability_stats.get("h24")
            d7 = availability_stats.get("d7")
            if isinstance(h24, (int, float)) or isinstance(d7, (int, float)):
                lines.append(
                    f"  可用率 24h {_fmt_rate(h24, short=True)} · "
                    f"7d {_fmt_rate(d7, short=True)}"
                )
            if isinstance(same_model, (int, float)) and same_model < 100:
                lines.append(f"  模型不匹配 {100 - same_model:.1f}%")
            if isinstance(intel, dict):
                swapped = intel.get("swapped")
                cut = intel.get("cut")
                if isinstance(swapped, (int, float)) and swapped:
                    lines.append(f"  路由切换 {swapped:.1f}%")
                if isinstance(cut, (int, float)) and cut:
                    lines.append(f"  提前截断 {cut:.1f}%")

            bars = _history_bars(model.get("cells"), data)
            if bars:
                if status == "down":
                    last_up = _last_healthy_time(bars, data)
                    if last_up is not None:
                        lines.append(f"  最近正常 {_fmt_time(last_up.isoformat())}")
                elif any(b == "down" for b in bars):
                    last_down_idx = max(i for i, b in enumerate(bars) if b == "down")
                    went_down = _cell_time(data, last_down_idx)
                    if went_down is not None:
                        lines.append(
                            f"  近 24h 曾中断（{_fmt_time(went_down.isoformat())}）"
                        )
                stripe = "".join(
                    {"ok": "🟩", "warn": "🟨", "down": "🟥", "unknown": "⬜"}[b]
                    for b in bars
                )
                lines.append(f"  近 24h（旧→新，每格 30 分钟）：\n  {stripe}")
        return "\n".join(lines)

    def _resolve_model_and_entry(
        self, data: dict, target: str
    ) -> tuple[str | None, list[tuple[str, dict, dict]] | None, str]:
        """Resolve user input of ``model[@cohort]`` against the payload.

        Args:
            data: Status payload.
            target: Raw user input, e.g. ``opus-5-5@paid``.

        Returns:
            ``(model_id, entries, "")`` on hit, ``(None, None, error)`` on miss.
        """
        model_part, cohort_part = _split_target(target)
        all_models = self._all_models(data)
        if not model_part:
            return None, None, "用法：/mirasim <模型>[@<free|paid|cloud>]"
        try:
            model_id = _resolve_id(
                list(all_models), model_part, "发送 /mirasim all 查看全部模型。"
            )
        except LookupError as exc:
            return None, None, str(exc)
        entries = all_models[model_id]
        if cohort_part is None:
            return model_id, entries, ""
        matched_cohort: str | None = None
        for cid, _m, _a in entries:
            if cid.lower() == cohort_part:
                matched_cohort = cid
                break
        if matched_cohort is None:
            for cid, name in COHORT_NAMES.items():
                if name == cohort_part and any(e[0] == cid for e in entries):
                    matched_cohort = cid
                    break
        if matched_cohort is None:
            return (
                None,
                None,
                (
                    f"未知池「{cohort_part}」，{model_id} 可用："
                    f"{'、'.join(f'{c}（{COHORT_NAMES.get(c, c)}）' for c, _m, _a in entries)}"
                ),
            )
        return model_id, [e for e in entries if e[0] == matched_cohort], ""

    async def _subscribe(self, umo: str, target: str) -> str:
        """Subscribe a session to a model, one of its cohorts, or all models.

        Args:
            umo: Unified message origin of the session.
            target: A model ID, ``model@cohort``, or one of ``ALL_WORDS``.

        Returns:
            The reply text.
        """
        if not target:
            return "用法：/mirasim sub <模型>[@<free|paid|cloud>] 或 /mirasim sub all"
        wanted = self._subs.get(umo, [])
        if target.lower() in ALL_WORDS:
            self._subs[umo] = [ALL_MODELS]
            await self.put_kv_data("subscriptions", self._subs)
            return "已订阅全部模型（包括之后新增的模型），状态变化时会推送到本会话。"
        if ALL_MODELS in wanted:
            return "本会话已订阅全部模型，无需单独订阅。"

        try:
            data = await self._fetch_status()
        except Exception as exc:
            logger.warning(f"[Mirasim] Status query failed: {exc!r}")
            return f"获取 Mirasim 状态失败，暂时无法确认模型：{str(exc) or type(exc).__name__}"
        model_id, entries, error = self._resolve_model_and_entry(data, target)
        if error:
            return error
        _model_part, cohort_part = _split_target(target)
        entry = f"{model_id}@{cohort_part}" if cohort_part else model_id
        if entry in wanted:
            return f"本会话已订阅 {entry}。"

        # Check whether an existing subscription already covers this target.
        sample_keys = [f"{model_id}@{c}" for c, _m, _a in entries]
        if any(_covers(e, k) for e in wanted for k in sample_keys):
            return "本会话的现有订阅已覆盖该目标。"

        self._subs[umo] = [*wanted, entry]
        await self.put_kv_data("subscriptions", self._subs)

        if cohort_part:
            cohort_id, model, _a = entries[0]
            status, _ = _now_status_of(model)
            return (
                f"已订阅 {model_id}@{cohort_id}（{COHORT_NAMES.get(cohort_id, cohort_id)}池，"
                f"当前 {STATUS_ICONS.get(status, '❔')} {STATUS_NAMES.get(status, status)}），"
                "状态变化时会推送到本会话。"
            )
        statuses = [_now_status_of(m)[0] for _c, m, _a in entries]
        worst = _worst_status(statuses)
        return (
            f"已订阅 {model_id} 的全部 {len(entries)} 个池"
            f"（当前最差 {STATUS_ICONS.get(worst, '❔')} {STATUS_NAMES.get(worst, worst)}），"
            "状态变化时会推送到本会话。"
        )

    async def _unsubscribe(self, umo: str, target: str) -> str:
        """Remove one model, one cohort, or everything from a session's subs.

        Args:
            umo: Unified message origin of the session.
            target: A subscribed model, ``model@cohort``, or ``all``.

        Returns:
            The reply text.
        """
        if not target:
            return (
                "用法：/mirasim unsub <模型>[@<free|paid|cloud>] 或 /mirasim unsub all"
            )
        wanted = self._subs.get(umo, [])
        if not wanted:
            return "本会话还没有订阅任何模型。"
        if target.lower() in ALL_WORDS:
            self._subs.pop(umo, None)
            await self.put_kv_data("subscriptions", self._subs)
            return "已取消本会话的全部订阅。"
        if ALL_MODELS in wanted:
            return "本会话订阅的是全部模型，发送 /mirasim unsub all 取消。"

        model_part, cohort_part = _split_target(target)
        matched = [
            e
            for e in wanted
            if _split_target(e)[0].lower() == model_part.lower()
            and (
                cohort_part is None
                or (_split_target(e)[1] or "") == cohort_part.lower()
            )
        ]
        if not matched:
            hint = "发送 /mirasim list 查看本会话的订阅。"
            try:
                data = await self._fetch_status()
            except Exception:
                return f"本会话没有订阅「{target}」，{hint}"
            try:
                resolved = _resolve_id(list(self._all_models(data)), model_part, hint)
                return f"本会话没有订阅 {resolved}，{hint}"
            except LookupError:
                return f"本会话没有订阅「{target}」，{hint}"

        remaining = [e for e in wanted if e not in matched]
        if remaining:
            self._subs[umo] = remaining
        else:
            self._subs.pop(umo, None)
        await self.put_kv_data("subscriptions", self._subs)
        return f"已取消订阅 {'、'.join(matched)}。"
