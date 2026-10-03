"""Mirasim model availability monitor: on-demand status and change pushes."""

from __future__ import annotations

import asyncio
import contextlib
import re
from datetime import datetime

import httpx

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star

DEFAULT_API_URL = "https://mirasim-status.521882.xyz/api/status"
REQUEST_TIMEOUT = 20
# Subscription entry meaning "every model", including models added later.
ALL_MODELS = "*"
ALL_WORDS = ("all", "*", "全部")
# Samples drawn in the single-model view, matching what fits on a phone line.
HISTORY_BAR_SAMPLES = 20

STATUS_ICONS = {"up": "✅", "down": "❌", "stale": "⏸️", "unknown": "❔"}
STATUS_NAMES = {"up": "可用", "down": "失败", "stale": "已过期", "unknown": "待检测"}
# Problems sort first so the top of the list is all a reader needs to check.
STATUS_ORDER = {"down": 0, "stale": 1, "unknown": 2, "up": 3}
# Error codes reported by the monitor, worded the way its status page does.
ERROR_NAMES = {
    "timeout": "检测超时",
    "network_error": "上游网络连接失败",
    "network_busy": "检测连接繁忙",
    "invalid_json": "上游返回无效 JSON",
    "body_too_large": "上游响应超出大小限制",
    "redirect_blocked": "上游重定向已阻止",
    "upstream_error": "上游返回错误",
    "upstream_blocked": "上游拒绝或拦截检测",
    "upstream_truncated": "上游输出被截断",
    "upstream_incomplete": "上游输出未完整结束",
    "upstream_empty_output": "上游未返回有效输出",
    "inventory_invalid": "目录格式无效",
    "inventory_empty": "目录为空",
    "inventory_error": "模型目录获取失败",
    "invalid_response": "上游响应无效",
    "empty_response": "上游返回为空",
    "internal_error": "检测器内部错误",
    "probe_error": "检测失败",
    "storage_error": "监控存储写入失败",
}

USAGE = (
    "Mirasim 模型状态\n"
    "/mirasim — 查看模型状态\n"
    "/mirasim all — 查看全部模型\n"
    "/mirasim <模型> — 查看单个模型详情\n"
    "/mirasim sub <模型|all> — 本会话订阅状态变化推送\n"
    "/mirasim unsub <模型|all> — 取消订阅\n"
    "/mirasim list — 查看本会话的订阅\n"
    "模型名可只写能唯一确定的一部分，如 opus-5-5。"
)


def _parse_time(value: object) -> datetime | None:
    """Parse an ISO 8601 timestamp from the monitor.

    Args:
        value: Raw field value, normally like ``2026-10-03T12:26:22.030Z``.

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
    """Format a monitor timestamp in local time, dropping the date for today.

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


def _fmt_latency(value: object) -> str:
    """Format a latency in milliseconds as seconds, e.g. ``3.4s``.

    Args:
        value: Latency in milliseconds.

    Returns:
        The formatted latency, or ``—`` when unknown.
    """
    if not isinstance(value, (int, float)) or value < 0:
        return "—"
    return f"{value / 1000:.1f}s"


def _error_text(code: object) -> str:
    """Translate a monitor error code into a readable reason.

    Args:
        code: The ``last_error`` value, e.g. ``upstream_truncated``.

    Returns:
        The reason text. Unknown codes collapse into a generic phrase because
        they may carry raw upstream content, which the status page also hides.
    """
    if not code:
        return "未知原因"
    if code in ERROR_NAMES:
        return ERROR_NAMES[code]
    if isinstance(code, str) and re.fullmatch(r"HTTP [1-5]\d{2}", code):
        return f"上游 {code}"
    return "检测失败"


def _effective_status(model: dict, data: dict) -> str:
    """Return the status the status page itself would show for a model.

    A model that was never checked is ``unknown``, and one whose last check is
    older than ``stale_after_seconds`` is ``stale`` whatever it said, because
    such a result no longer proves anything about the present.

    Args:
        model: One entry of the payload's ``models``.
        data: The whole payload, for the server clock and staleness limit.

    Returns:
        One of ``up``, ``down``, ``stale`` and ``unknown``.
    """
    checked = _parse_time(model.get("last_checked_at"))
    if checked is None:
        return "unknown"
    status = model.get("status")
    now = _parse_time(data.get("now"))
    stale_after = data.get("stale_after_seconds")
    if status == "stale" or (
        now is not None
        and isinstance(stale_after, (int, float))
        and (now - checked).total_seconds() > stale_after
    ):
        return "stale"
    return status if status in ("up", "down") else "unknown"


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


class MirasimStatus(Star):
    """Show Mirasim model availability and push confirmed status changes."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        # Session (unified_msg_origin) -> subscribed model IDs or [ALL_MODELS].
        self._subs: dict[str, list[str]] = {}
        # Model ID -> last confirmed "up"/"down", the baseline for pushes.
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
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("unexpected status payload")
        return data

    async def _poll_loop(self) -> None:
        """Poll the monitor forever and push confirmed status changes."""
        while True:
            # Sleep first: right after startup the platform adapters may still
            # be connecting, and a push sent then would be lost.
            interval = max(30, int(self.config.get("poll_interval", 60) or 60))
            await asyncio.sleep(interval)
            try:
                await self._check_changes(await self._fetch_status())
            except Exception as exc:
                logger.warning(f"[Mirasim] Status poll failed: {exc!r}")

    async def _check_changes(self, data: dict) -> None:
        """Detect confirmed up/down transitions and push them to subscribers.

        A model changes state only once its latest ``confirm_samples`` samples
        agree, which filters out single-sample blips. The first state seen for
        a model is recorded silently as its baseline. Every session receives
        one message per poll, since related models tend to fail together.

        Args:
            data: Status payload returned by :meth:`_fetch_status`.
        """
        confirm = max(1, int(self.config.get("confirm_samples", 2) or 1))
        changes: list[tuple[str, str]] = []
        dirty = False
        for model in data["models"]:
            model_id = model.get("id")
            if not model_id or not model.get("active"):
                continue
            history = sorted(model.get("history") or [], key=lambda s: s.get("at", ""))
            recent = {sample.get("status") for sample in history[-confirm:]}
            if len(history) < confirm or len(recent) != 1:
                continue
            status = recent.pop()
            previous = self._states.get(model_id)
            if status not in ("up", "down") or status == previous:
                continue
            self._states[model_id] = status
            dirty = True
            if previous is None:
                continue

            # The current run of `status` samples starts where it really began.
            start = len(history)
            while start > 0 and history[start - 1].get("status") == status:
                start -= 1
            since = history[start].get("at")
            if status == "down":
                reason = _error_text(model.get("last_error"))
                line = f"🔴 {model_id} 故障：{reason}（{_fmt_time(since)} 起）"
            else:
                line = (
                    f"🟢 {model_id} 已恢复，当前耗时 "
                    f"{_fmt_latency(model.get('latency_ms'))}"
                )
                # The failure run right before the recovery gives the outage.
                outage_start = start
                while (
                    outage_start > 0
                    and history[outage_start - 1].get("status") != status
                ):
                    outage_start -= 1
                began = _parse_time(history[outage_start].get("at"))
                ended = _parse_time(since)
                if outage_start < start and began and ended:
                    # A run reaching the oldest sample began before the window.
                    prefix = "超过" if outage_start == 0 else "约"
                    outage = _fmt_duration((ended - began).total_seconds())
                    line += f"，故障持续{prefix} {outage}"
            changes.append((model_id, line))

        if dirty:
            await self.put_kv_data("states", self._states)
        if not changes:
            return
        logger.info(
            f"[Mirasim] Status changes: {[model_id for model_id, _ in changes]}"
        )

        # Snapshot the sessions: a command may edit them while we await sends.
        for umo, wanted in list(self._subs.items()):
            lines = [
                line
                for model_id, line in changes
                if ALL_MODELS in wanted or model_id in wanted
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
                text = f"本会话订阅了 {len(wanted)} 个模型：\n" + "\n".join(
                    f"· {model_id}" for model_id in wanted
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
                    text = self._render_overview(data, show_all=bool(act))
        yield event.plain_result(text)

    async def _subscribe(self, umo: str, target: str) -> str:
        """Subscribe a session to one model or to every model.

        Args:
            umo: Unified message origin of the session.
            target: A model ID, a unique part of one, or one of ``ALL_WORDS``.

        Returns:
            The reply text.
        """
        if not target:
            return "用法：/mirasim sub <模型|all>"
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
        models = {
            model["id"]: model
            for model in data["models"]
            if model.get("id") and model.get("active")
        }
        try:
            model_id = _resolve_id(
                list(models), target, "发送 /mirasim all 查看全部模型。"
            )
        except LookupError as exc:
            return str(exc)
        if model_id in wanted:
            return f"本会话已订阅 {model_id}。"

        self._subs[umo] = [*wanted, model_id]
        await self.put_kv_data("subscriptions", self._subs)
        status = _effective_status(models[model_id], data)
        return (
            f"已订阅 {model_id}（当前 {STATUS_ICONS[status]} {STATUS_NAMES[status]}），"
            "状态变化时会推送到本会话。"
        )

    async def _unsubscribe(self, umo: str, target: str) -> str:
        """Remove one model, or every model, from a session's subscriptions.

        Args:
            umo: Unified message origin of the session.
            target: A subscribed model ID, a unique part of one, or ``all``.

        Returns:
            The reply text.
        """
        if not target:
            return "用法：/mirasim unsub <模型|all>"
        wanted = self._subs.get(umo, [])
        if not wanted:
            return "本会话还没有订阅任何模型。"
        if target.lower() in ALL_WORDS:
            self._subs.pop(umo, None)
            await self.put_kv_data("subscriptions", self._subs)
            return "已取消本会话的全部订阅。"
        if ALL_MODELS in wanted:
            return "本会话订阅的是全部模型，发送 /mirasim unsub all 取消。"

        try:
            model_id = _resolve_id(
                wanted, target, "发送 /mirasim list 查看本会话的订阅。"
            )
        except LookupError as exc:
            return str(exc)
        remaining = [subscribed for subscribed in wanted if subscribed != model_id]
        if remaining:
            self._subs[umo] = remaining
        else:
            self._subs.pop(umo, None)
        await self.put_kv_data("subscriptions", self._subs)
        return f"已取消订阅 {model_id}。"

    def _render_overview(self, data: dict, show_all: bool) -> str:
        """Render the status list of the configured models, or of all of them.

        Args:
            data: Status payload.
            show_all: Ignore the ``display_models`` setting.

        Returns:
            The reply text.
        """
        models = [
            model for model in data["models"] if model.get("id") and model.get("active")
        ]
        configured = [
            str(item).strip()
            for item in self.config.get("display_models") or []
            if str(item).strip()
        ]
        filtered = bool(configured) and not show_all
        missing: list[str] = []
        if filtered:
            by_id = {model["id"].lower(): model for model in models}
            picked = []
            for model_id in dict.fromkeys(configured):
                if model_id.lower() in by_id:
                    picked.append(by_id[model_id.lower()])
                else:
                    missing.append(model_id)
            models = picked

        rows = sorted(
            ((_effective_status(model, data), model) for model in models),
            key=lambda row: (STATUS_ORDER[row[0]], row[1]["id"]),
        )
        up = sum(1 for status, _ in rows if status == "up")
        lines = [f"Mirasim 模型状态 · 可用 {up}/{len(rows)}"]
        for status, model in rows:
            if status == "up":
                rate = model.get("success_rate_24h")
                rate_text = (
                    f"{round(rate, 1):g}%"
                    if isinstance(rate, (int, float)) and model.get("samples_24h")
                    else "—"
                )
                detail = f"{_fmt_latency(model.get('latency_ms'))} · 24h {rate_text}"
            elif status == "down":
                detail = _error_text(model.get("last_error"))
            else:
                detail = STATUS_NAMES[status]
            lines.append(f"{STATUS_ICONS[status]} {model['id']} · {detail}")
        if not rows:
            lines.append("暂无模型数据。")
        if missing:
            lines.append(f"未找到：{'、'.join(missing)}")

        inventory_error = (data.get("inventory") or {}).get("error")
        if inventory_error:
            lines.append(f"⚠️ 模型目录更新失败：{_error_text(inventory_error)}")
        footer = f"检测于 {_fmt_time((data.get('scan') or {}).get('last_finished_at'))}"
        interval = data.get("interval_seconds")
        if isinstance(interval, (int, float)) and interval > 0:
            footer += f" · 每 {_fmt_duration(interval)}一轮"
        lines.append(footer)
        if filtered:
            lines.append("仅展示配置的模型，发送 /mirasim all 查看全部。")
        return "\n".join(lines)

    def _render_model(self, data: dict, query: str) -> str:
        """Render the detail view of one model.

        Args:
            data: Status payload.
            query: The model the user asked for.

        Returns:
            The reply text.
        """
        # Models that left the inventory stay queryable for their history.
        models = {model["id"]: model for model in data["models"] if model.get("id")}
        try:
            model_id = _resolve_id(
                list(models), query, "发送 /mirasim all 查看全部模型。"
            )
        except LookupError as exc:
            return str(exc)
        model = models[model_id]
        status = _effective_status(model, data)

        lines = [f"{STATUS_ICONS[status]} {model_id} · {STATUS_NAMES[status]}"]
        if not model.get("active"):
            lines.append("该模型已退出监控目录，以下仅为历史记录。")
        if status == "down":
            lines.append(f"失败原因：{_error_text(model.get('last_error'))}")
        elif status == "stale":
            lines.append("检测结果已过期，当前是否可用无法确认。")
        lines.append(
            f"响应耗时：{_fmt_latency(model.get('latency_ms'))}"
            f"（24h 平均 {_fmt_latency(model.get('avg_latency_ms_24h'))}）"
        )
        if status == "up" and model.get("current_stable_since"):
            lines.append(
                f"连续可用：{_fmt_duration(model.get('stable_seconds') or 0)}"
                f"（{_fmt_time(model['current_stable_since'])} 起）"
            )
        rate = model.get("success_rate_24h")
        samples = model.get("samples_24h") or 0
        rate_text = (
            f"{round(rate, 1):g}%"
            if isinstance(rate, (int, float)) and samples
            else "—"
        )
        lines.append(f"24h 成功率：{rate_text}（{samples} 次采样）")
        lines.append(f"最近检测：{_fmt_time(model.get('last_checked_at'))}")
        if status != "up" and model.get("last_success_at"):
            lines.append(f"最近成功：{_fmt_time(model['last_success_at'])}")
        if model.get("last_failure_at"):
            lines.append(f"最近失败：{_fmt_time(model['last_failure_at'])}")

        history = sorted(model.get("history") or [], key=lambda s: s.get("at", ""))
        recent = history[-HISTORY_BAR_SAMPLES:]
        if recent:
            bar = "".join(
                "🟩" if sample.get("status") == "up" else "🟥" for sample in recent
            )
            lines.append(f"最近 {len(recent)} 次采样（旧→新）：\n{bar}")
        return "\n".join(lines)
