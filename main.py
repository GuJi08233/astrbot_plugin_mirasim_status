"""Mirasim status monitor: on-demand status cards and confirmed change pushes.

The status payload (schema 2) is organised as cohorts -> agents -> models. A
model's ``id`` is only unique *inside one agent* (``m1``..``m6`` repeat in every
agent and cohort), while its ``name`` is the real identity (``opus 5.5``), so
every model here is keyed by a slug of that name. Statuses come straight from
the payload's ``now.status`` and its own ``thresholds`` block.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star

from . import pricing

try:  # AstrBot's own commands annotate trailing params with this marker class.
    from astrbot.core.star.filter.command import GreedyStr
except Exception:  # pragma: no cover - very old AstrBot without the marker
    GreedyStr = str

DEFAULT_API_URL = "https://mirasim.ai/api/status"
REQUEST_TIMEOUT = 20
# mirasim.ai refreshes its aggregation once a minute; a faster poll finds nothing.
MIN_POLL_INTERVAL = 60
DEFAULT_POLL_INTERVAL = 60
# One change of mind is enough at the official minute cadence; two keeps blips quiet.
DEFAULT_CONFIRM_POLLS = 2
# Fallbacks used only when a payload arrives without its own thresholds block.
DEFAULT_THRESHOLD_GOOD = 99.0
DEFAULT_THRESHOLD_WARN = 95.0
# Reuse a just-fetched payload for this long so one burst of commands costs one request.
CACHE_TTL = 10
# Subscription entry meaning "every cohort of every model", including later models.
ALL_MODELS = "*"
ALL_WORDS = ("all", "*", "全部")
COHORT_ORDER = ("free", "paid", "cloud")
COHORT_NAMES = {"free": "免费", "paid": "订阅", "cloud": "云端"}
# What a user may type for a cohort: the ID, or the Chinese label.
COHORT_ALIASES = {
    "free": "free",
    "paid": "paid",
    "cloud": "cloud",
    "免费": "free",
    "体验": "free",
    "订阅": "paid",
    "云端": "cloud",
}
TEMPLATE_PATH = Path(__file__).parent / "templates" / "status.html"
# PNG keeps the small text crisp; "ultra" is a 1.8x device pixel ratio on the
# official T2I service, which phones need once they scale the card down.
RENDER_OPTIONS = {
    "type": "png",
    "full_page": True,
    "device_scale_factor_level": "ultra",
}

STATUS_ICONS = {
    "ok": "✅",
    "degraded": "⚠️",
    "down": "❌",
    "nodata": "❔",
    "unknown": "❔",
}
STATUS_NAMES = {
    "ok": "正常",
    "degraded": "降级",
    "down": "异常",
    "nodata": "无数据",
    "unknown": "未知",
}
# Problems sort first so the top of the list is all a reader needs to check.
STATUS_ORDER = {"down": 0, "degraded": 1, "nodata": 2, "unknown": 2, "ok": 3}
CHANGE_ICONS = {"ok": "🟢", "degraded": "🟡", "down": "🔴", "nodata": "⚪"}
STRIPE_ICONS = {"ok": "🟩", "degraded": "🟨", "down": "🟥", "unknown": "⬜"}
# The payload spells statuses its own way; accept the common variants too.
STATUS_ALIASES = {
    "ok": "ok",
    "good": "ok",
    "healthy": "ok",
    "up": "ok",
    "degraded": "degraded",
    "warn": "degraded",
    "warning": "degraded",
    "unstable": "degraded",
    "down": "down",
    "bad": "down",
    "error": "down",
    "outage": "down",
    "nodata": "nodata",
    "no_data": "nodata",
    "none": "nodata",
    "unknown": "unknown",
}
# Failure classes reported at the agent level, worded the way the site does.
REASON_NAMES = {
    "throttle": "限流",
    "rate_limit": "限流",
    "outage": "故障",
    "capacity": "容量不足",
    "overload": "过载",
    "error": "错误",
    "latency": "延迟",
    "quality": "质量",
}

USAGE = (
    "Mirasim 模型状态\n"
    "/mirasim — 查看模型状态（默认图片卡片）\n"
    "/mirasim all — 查看全部模型\n"
    "/mirasim <模型> — 查看单个模型各池详情\n"
    "/mirasim <模型>@<free|paid|cloud> — 只看指定池\n"
    "/mirasim agents — 按厂商查看整体状态\n"
    "/mirasim price — 查看模型价目表\n"
    "/mirasim sub <模型>[@池]|all — 本会话订阅状态变化推送\n"
    "/mirasim unsub <模型>[@池]|all — 取消订阅\n"
    "/mirasim list — 查看本会话的订阅\n"
    "/mirasim help — 显示本说明\n"
    "模型名可只写能唯一确定的一部分，如 opus-5-5、gpt 6 astra。"
)


def _slug(text: object) -> str:
    """Turn a display name into a stable lowercase key.

    Args:
        text: Anything name-like, e.g. ``Gemini 4 Argon`` or ``gpt-6.1-sol``.

    Returns:
        Letters and digits joined by single dashes, e.g. ``gemini-4-argon``.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower())).strip("-")


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
        return f"{days}天{hours}小时" if hours else f"{days}天"
    if hours:
        return f"{hours}小时{minutes}分钟" if minutes else f"{hours}小时"
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
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
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
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{round(value, 1):g}%" if short else f"{value:.1f}%"


def _fmt_price(value: object) -> str:
    """Format a price in credits per million tokens.

    Args:
        value: The price as a number.

    Returns:
        The price without a trailing ``.0``, or an empty string when the
        price page has none.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ""
    return f"{value:g}"


def _price_bits(price: pricing.Price) -> str:
    """Render one model's three prices.

    Args:
        price: The price entry.

    Returns:
        The input, output and cache read prices in one line.
    """
    parts = []
    for label, (list_price, current) in (
        ("输入", price.input_price),
        ("输出", price.output_price),
        ("缓存读取", price.cache_read_price),
    ):
        charged = _fmt_price(current)
        if not charged:
            parts.append(f"{label} —")
            continue
        struck = _fmt_price(list_price)
        if struck and struck != charged:
            parts.append(f"{label} {charged}（刊例 {struck}）")
        else:
            parts.append(f"{label} {charged}")
    return " · ".join(parts)


WINDOW_UNITS = {"m": "分钟", "h": "小时", "d": "天"}


def _fmt_window(value: object) -> str:
    """Format the aggregation window the site reports, e.g. ``15m`` -> ``15分钟``.

    Args:
        value: Raw ``window`` field; ``null`` on most model entries.

    Returns:
        Human text, or an empty string when unknown.
    """
    if not isinstance(value, str) or not value:
        return ""
    match = re.fullmatch(r"\s*(\d+)\s*([mhd])\s*", value.lower())
    if not match:
        return value
    amount, unit = int(match.group(1)), match.group(2)
    return f"{amount}{WINDOW_UNITS[unit]}"


def _thresholds(data: object) -> tuple[float, float, int | None]:
    """Read the site's own availability thresholds.

    The payload ships ``thresholds: {good, warn, minTurns}``; using it keeps the
    plugin's wording in step with the status page instead of guessing.

    Args:
        data: Status payload.

    Returns:
        ``(good, warn, min_turns)`` with fallbacks for missing or inverted values.
    """
    raw = data.get("thresholds") if isinstance(data, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    good = raw.get("good")
    warn = raw.get("warn")
    min_turns = raw.get("minTurns")
    good = (
        float(good)
        if isinstance(good, (int, float)) and not isinstance(good, bool)
        else DEFAULT_THRESHOLD_GOOD
    )
    warn = (
        float(warn)
        if isinstance(warn, (int, float)) and not isinstance(warn, bool)
        else DEFAULT_THRESHOLD_WARN
    )
    if warn > good:
        warn = good
    min_turns = (
        min_turns
        if isinstance(min_turns, int) and not isinstance(min_turns, bool)
        else None
    )
    return good, warn, min_turns


def _status_from_availability(availability: object, good: float, warn: float) -> str:
    """Bucket an availability percentage against the site's thresholds.

    Args:
        availability: Percentage as a number.
        good: At or above this the model counts as 正常.
        warn: At or above this it counts as 降级.

    Returns:
        ``ok``, ``degraded``, ``down`` or ``nodata``.
    """
    if not isinstance(availability, (int, float)) or isinstance(availability, bool):
        return "nodata"
    if availability >= good:
        return "ok"
    if availability >= warn:
        return "degraded"
    return "down"


def _model_status(model: dict, data: object = None) -> tuple[str, float | None]:
    """Read a model's current status and availability.

    ``now.status`` is authoritative because the site weighs turn counts on top
    of the percentage; availability only decides the bucket when the status is
    missing or in an unknown vocabulary.

    Args:
        model: Model dict from the payload.
        data: Whole payload, for its thresholds block.

    Returns:
        ``(status, availability)`` where status is ok/degraded/down/nodata.
    """
    now = model.get("now") if isinstance(model, dict) else None
    now = now if isinstance(now, dict) else {}
    availability = now.get("availability")
    raw_status = now.get("status")
    if isinstance(raw_status, str):
        status = STATUS_ALIASES.get(raw_status.strip().lower())
        if status:
            return status, availability if isinstance(
                availability, (int, float)
            ) else None
    good, warn, _min_turns = _thresholds(data)
    return _status_from_availability(availability, good, warn), (
        availability if isinstance(availability, (int, float)) else None
    )


def _cell_status(value: object, good: float, warn: float) -> str:
    """Translate one history cell to a colour bucket.

    Cells are stored as availability ×10 (1000 = 100%), and -1 marks a gap.

    Args:
        value: Raw cell value.
        good: Availability percentage still counted as 正常.
        warn: Availability percentage still counted as 降级.

    Returns:
        One of ``ok``, ``degraded``, ``down``, ``unknown``.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return "unknown"
    return _status_from_availability(value / 10.0, good, warn)


def _worst_status(statuses: list[str]) -> str:
    """Pick the worst of several status labels.

    Args:
        statuses: Labels to compare.

    Returns:
        The worst one under STATUS_ORDER.
    """
    return min(statuses, key=lambda s: STATUS_ORDER.get(s, 99), default="unknown")


def _history_bars(cells: object, data: object = None) -> list[str]:
    """Build the per-cell status list used by stripes and outage bounds.

    Args:
        cells: Raw ``cells`` array of a model.
        data: Whole payload, for its thresholds block.

    Returns:
        One ``ok``/``degraded``/``down``/``unknown`` per cell, oldest first.
    """
    if not isinstance(cells, list):
        return []
    good, warn, _min_turns = _thresholds(data)
    return [_cell_status(value, good, warn) for value in cells]


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
        if bars[i] in ("ok", "degraded"):
            return _cell_time(data, i)
    return None


def _last_streak(bars: list[str], targets: tuple[str, ...]) -> tuple[int, int] | None:
    """Locate the newest run of cells whose status is in ``targets``.

    Args:
        bars: Per-cell statuses ordered oldest first.
        targets: Statuses that count as part of the run.

    Returns:
        ``(start_index, length)`` of the trailing run, or None when there is none.
    """
    end = len(bars) - 1
    while end >= 0 and bars[end] not in targets:
        end -= 1
    if end < 0:
        return None
    start = end
    while start >= 0 and bars[start] in targets:
        start -= 1
    start += 1
    return start, end - start + 1


def _resolve_key(
    keys: list[str], query: str, hint: str, aliases: dict[str, str] | None = None
) -> str:
    """Resolve user input to exactly one model key.

    A slug-exact match wins; otherwise the input must be part of exactly one
    key, so ``opus-5-5`` and ``opus 5.5`` both find ``opus-5-5``.

    Args:
        keys: Candidate model keys (already slugged).
        query: What the user typed.
        hint: Appended to the "not found" message to point at the next step.
        aliases: Legacy spellings mapped onto current keys, see
            ``_model_aliases``. Matched exactly, after the current keys, so a
            name that is already valid keeps winning.

    Returns:
        The matched key.

    Raises:
        LookupError: Nothing or several keys match; the message is user-facing.
    """
    needle = _slug(query)
    if not needle:
        raise LookupError(f"没有找到匹配「{query}」的模型，{hint}")
    for key in keys:
        if key == needle:
            return key
    if aliases and needle in aliases:
        return aliases[needle]
    hits = sorted(key for key in keys if needle in key)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise LookupError(f"没有找到匹配「{query}」的模型，{hint}")
    raise LookupError(
        f"「{query}」匹配到多个模型：{'、'.join(hits)}，请输入更完整的名称。"
    )


def _split_target(text: str) -> tuple[str, str | None]:
    """Split ``model@cohort`` into its two parts.

    Args:
        text: Raw target from the user.

    Returns:
        ``(model_part, cohort_part or None)``; the cohort is normalised to its ID.
    """
    if "@" not in text:
        return text.strip(), None
    model_part, cohort_part = text.rsplit("@", 1)
    raw_cohort = cohort_part.strip().lower()
    if not raw_cohort:
        return model_part.strip(), None
    return model_part.strip(), COHORT_ALIASES.get(raw_cohort, raw_cohort)


def _covers(entry: str, key: str, aliases: dict[str, str] | None = None) -> bool:
    """Tell whether a subscription entry matches a ``model@cohort`` change key.

    Args:
        entry: Stored subscription, ``model``, ``model@cohort`` or ``*``.
        key: Change key, always ``model@cohort``.
        aliases: Legacy spellings mapped onto current keys, see
            ``_model_aliases``. A subscription stored under the old
            vendor-prefixed name still covers the model it names today.

    Returns:
        True when the entry wants this key.
    """
    if entry == ALL_MODELS:
        return True
    model_part, cohort_part = _split_target(entry)
    key_model, key_cohort = _split_target(key)
    if key_model != model_part and (
        not aliases or aliases.get(model_part) != key_model
    ):
        return False
    return cohort_part is None or cohort_part == key_cohort


def _format_reasons(reasons: object) -> str:
    """Render the agent-level failure breakdown as bare text.

    Args:
        reasons: ``reasons`` list from the agent blob.

    Returns:
        Text such as ``限流79%·故障21%``, or an empty string when there is none.
        Callers add their own brackets so the result reads well inline.
    """
    if not isinstance(reasons, list):
        return ""
    parts = []
    for item in reasons:
        if not isinstance(item, dict):
            continue
        cls = item.get("class")
        share = item.get("share")
        if not cls or not isinstance(share, (int, float)) or isinstance(share, bool):
            continue
        if share < 1:
            continue
        label = REASON_NAMES.get(str(cls), str(cls))
        parts.append(f"{label}{_fmt_share(share)}%")
    return "·".join(parts)


def _fmt_share(value: float) -> str:
    """Round a percentage share to a whole number, halves going up.

    ``round()`` and ``f"{x:.0f}"`` both round halves to even, so 78.5 would
    print as ``78`` while the monitor shows ``79``. The API ships whole numbers
    today (100, 98, 2); this keeps any future decimal sane.

    Args:
        value: Share such as ``100`` or ``21.4``.

    Returns:
        ``100`` or ``21``.
    """
    return f"{math.floor(float(value) + 0.5)}"


def _reason_note(reasons: object) -> str:
    """Wrap :func:`_format_reasons` in brackets for inline use.

    Args:
        reasons: ``reasons`` list from the agent blob.

    Returns:
        ``(限流79%·故障21%)``, or an empty string when there is none.
    """
    text = _format_reasons(reasons)
    return f"（{text}）" if text else ""


def _collect_models(data: object) -> dict[str, dict]:
    """Index every model by its display-name slug, with its per-cohort entries.

    Args:
        data: Status payload.

    Returns:
        ``{key: {"key", "name", "agent", "agent_id", "entries"}}`` in payload
        order; ``entries`` holds ``(cohort_id, model_dict, agent_dict)`` sorted
        by COHORT_ORDER.
    """
    index: dict[str, dict] = {}
    cohorts = data.get("cohorts") if isinstance(data, dict) else None
    for cohort in cohorts or []:
        cohort_id = cohort.get("id")
        if not cohort_id:
            continue
        for agent in cohort.get("agents") or []:
            for model in agent.get("models") or []:
                key = _slug(model.get("name")) or _slug(model.get("id"))
                if not key:
                    continue
                record = index.get(key)
                if record is None:
                    record = {
                        "key": key,
                        "name": model.get("name") or key,
                        "agent": agent.get("name") or agent.get("id") or "",
                        "agent_id": agent.get("id") or "",
                        "entries": [],
                    }
                    index[key] = record
                record["entries"].append((cohort_id, model, agent))
    order = {cid: i for i, cid in enumerate(COHORT_ORDER)}
    for record in index.values():
        record["entries"].sort(key=lambda e: order.get(e[0], len(COHORT_ORDER)))
    return index


def _model_aliases(models: dict[str, dict]) -> dict[str, str]:
    """Map the vendor-prefixed model names onto the current keys.

    A model used to be named after its vendor and its model, ``claude-opus-5-5``.
    The payload now repeats a short id inside every agent, so the model name
    alone is the identity and the key is ``opus-5-5``. Keeping the old
    spelling working means a stored ``display_models`` entry or a
    subscription survives the switch instead of silently going cold.

    Args:
        models: Model index from ``_collect_models``.

    Returns:
        ``{alias_slug: model_key}`` for every vendor-prefixed spelling.
    """
    aliases: dict[str, str] = {}
    for key, record in models.items():
        for prefix in (record.get("agent"), record.get("agent_id")):
            slug = _slug(prefix)
            if not slug:
                continue
            alias = _slug(f"{slug}-{record['name']}")
            if alias and alias != key:
                aliases[alias] = key
    return aliases


def _match_keys(
    models: dict[str, dict], query: str, aliases: dict[str, str]
) -> list[str]:
    """Return the model keys a configured entry selects.

    A model name selects one model. A vendor name selects every model of
    that vendor, because the keys carry the model name alone and so
    ``gpt`` or ``deepseek`` names no key of its own.

    Args:
        models: Model index from ``_collect_models``.
        query: The model or vendor the entry names.
        aliases: Legacy spellings mapped onto current keys.

    Returns:
        The matching keys, empty when the entry names nothing.

    Raises:
        LookupError: The query is part of several model names; the message is
            user-facing.
    """
    needle = _slug(query)
    if not needle:
        return []
    for key in models:
        if key == needle:
            return [key]
    if needle in aliases:
        return [aliases[needle]]
    vendor = [
        key
        for key, record in models.items()
        if needle in (_slug(record.get("agent")), _slug(record.get("agent_id")))
    ]
    if vendor:
        return vendor
    hits = sorted(key for key in models if needle in key)
    if len(hits) == 1:
        return hits
    if hits:
        raise LookupError(
            f"「{query}」匹配到多个模型：{'、'.join(hits)}，请输入更完整的名称。"
        )
    return []


def _collect_agents(data: object) -> dict[str, dict]:
    """Index every agent by ID with its per-cohort blobs.

    Args:
        data: Status payload.

    Returns:
        ``{agent_id: {"agent_id", "name", "entries"}}`` in payload order;
        ``entries`` holds ``(cohort_id, agent_dict)`` sorted by COHORT_ORDER.
    """
    index: dict[str, dict] = {}
    cohorts = data.get("cohorts") if isinstance(data, dict) else None
    for cohort in cohorts or []:
        cohort_id = cohort.get("id")
        if not cohort_id:
            continue
        for agent in cohort.get("agents") or []:
            agent_id = agent.get("id") or agent.get("name")
            if not agent_id:
                continue
            record = index.setdefault(
                agent_id,
                {
                    "agent_id": agent_id,
                    "name": agent.get("name") or agent_id,
                    "entries": [],
                },
            )
            record["entries"].append((cohort_id, agent))
    order = {cid: i for i, cid in enumerate(COHORT_ORDER)}
    for record in index.values():
        record["entries"].sort(key=lambda e: order.get(e[0], len(COHORT_ORDER)))
    return index


def _cohort_state(data: object) -> dict[str, str]:
    """Read each cohort's own rolled-up state.

    Args:
        data: Status payload.

    Returns:
        ``{cohort_id: state}`` using the same status vocabulary as models.
    """
    states: dict[str, str] = {}
    cohorts = data.get("cohorts") if isinstance(data, dict) else None
    for cohort in cohorts or []:
        cohort_id = cohort.get("id")
        raw = cohort.get("state")
        if not cohort_id or not isinstance(raw, str):
            continue
        states[cohort_id] = STATUS_ALIASES.get(raw.strip().lower(), "unknown")
    return states


def _agent_status(agent: dict, data: object = None) -> tuple[str, float | None]:
    """Read an agent's rolled-up status from its ``summary`` block.

    Args:
        agent: Agent dict from the payload.
        data: Whole payload, for its thresholds block.

    Returns:
        ``(status, availability)``, falling back to the agent's first model.
    """
    summary = agent.get("summary") if isinstance(agent, dict) else None
    summary = summary if isinstance(summary, dict) else {}
    if summary:
        status, availability = _model_status(summary, data)
        if status != "nodata":
            return status, availability
    for model in agent.get("models") or []:
        status, availability = _model_status(model, data)
        if status != "nodata":
            return status, availability
    return "nodata", None


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
        # (monotonic timestamp, payload) reused for CACHE_TTL seconds.
        self._cache: tuple[float, dict] | None = None
        # (monotonic timestamp, price list) reused for the price cache window.
        self._price_cache: tuple[float, pricing.PriceList] | None = None

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

    async def _fetch_status(self, *, use_cache: bool = True) -> dict:
        """Fetch the latest snapshot from the monitor.

        Args:
            use_cache: Reuse a payload fetched within CACHE_TTL seconds.

        Returns:
            The decoded status payload.

        Raises:
            httpx.HTTPError: The request failed or returned an error status.
            ValueError: The body is not a status document.
        """
        now = time.monotonic()
        if use_cache and self._cache is not None:
            cached_at, cached = self._cache
            if now - cached_at < CACHE_TTL:
                return cached
        url = str(self.config.get("api_url") or DEFAULT_API_URL).strip()
        # httpx honours the proxy environment AstrBot sets from its settings.
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("cohorts"), list):
            raise ValueError("unexpected status payload")
        self._cache = (now, data)
        return data

    async def _fetch_prices(self) -> pricing.PriceList:
        """Fetch the price list the site publishes.

        The prices live on a rendered page rather than in the status payload,
        and they change rarely, so the list is cached for hours. A failure
        keeps the last list that was read, so a temporary outage costs nothing
        but a stale price.

        Returns:
            The price list, empty when it was never read.
        """
        now = time.monotonic()
        if self._price_cache is not None:
            cached_at, cached = self._price_cache
            if now - cached_at < pricing.PRICE_CACHE_TTL:
                return cached
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.get(
                    pricing.PRICE_URL, headers={"Accept": "text/html"}
                )
                resp.raise_for_status()
            parsed = pricing.parse_prices(resp.text)
        except Exception as exc:
            logger.warning(f"[Mirasim] Price list query failed: {exc!r}")
            if self._price_cache is not None:
                return self._price_cache[1]
            return pricing.PriceList(entries={}, updated="")
        self._price_cache = (now, parsed)
        return parsed

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
                data = await self._fetch_status(use_cache=False)
                await self._check_changes(data, prev_data)
                prev_data = data
            except Exception as exc:
                logger.warning(f"[Mirasim] Status poll failed: {exc!r}")

    async def _check_changes(self, data: dict, prev_data: dict | None = None) -> None:
        """Detect confirmed transitions and push them to subscribers.

        A transition is confirmed when the model's status has matched
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

        grouped_now = _collect_models(data)
        grouped_prev = _collect_models(prev_data) if prev_data else {}

        # First poll ever: just record the baseline, no pushes.
        if not self._states:
            self._states = {
                f"{key}@{cohort_id}": _model_status(model, data)[0]
                for key, record in grouped_now.items()
                for cohort_id, model, _agent in record["entries"]
            }
            await self.put_kv_data("states", self._states)
            return

        changes: list[tuple[str, str, str, dict, dict, str]] = []
        dirty = False
        for key, record in grouped_now.items():
            prev_record = grouped_prev.get(key)
            prev_map = (
                {cid: model for cid, model, _a in prev_record["entries"]}
                if prev_record
                else {}
            )
            for cohort_id, model, agent in record["entries"]:
                change_key = f"{key}@{cohort_id}"
                current, _availability = _model_status(model, data)
                previous_confirmed = self._states.get(change_key, current)

                # With confirm >= 2, the previous poll's status must also have
                # equalled current; otherwise the change hasn't settled.
                if confirm >= 2:
                    prev_model = prev_map.get(cohort_id)
                    previous_now = (
                        current
                        if prev_model is None
                        else _model_status(prev_model, prev_data)[0]
                    )
                    if previous_now != current:
                        continue  # still flickering

                if previous_confirmed == current:
                    continue
                changes.append(
                    (
                        change_key,
                        previous_confirmed,
                        current,
                        model,
                        cohort_id,
                        _format_reasons(agent.get("reasons")),
                    )
                )
                self._states[change_key] = current
                dirty = True

        if not changes:
            return
        if dirty:
            await self.put_kv_data("states", self._states)
        logger.info(f"[Mirasim] Status changes: {[k for k, *_ in changes]}")

        lines_by_key: list[tuple[str, str]] = []
        for change_key, _old, new_status, model, cohort_id, reason in changes:
            lines_by_key.append(
                (
                    change_key,
                    _change_line(
                        new_status, model, cohort_id, reason, data, grouped_now
                    ),
                )
            )

        # Snapshot the sessions: a command may edit them while we await sends.
        aliases = _model_aliases(grouped_now)
        for umo, wanted in list(self._subs.items()):
            lines = [
                line
                for key, line in lines_by_key
                if any(_covers(e, key, aliases) for e in wanted)
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
        target: GreedyStr = "",
    ):
        """查看 Mirasim 模型状态，或管理本会话的状态推送订阅。"""
        act = action.strip().lower()
        rest = target.strip()
        umo = event.unified_msg_origin
        # Model names contain spaces ("opus 5.5"), so the whole argument tail is
        # one string: a known verb takes it as its target, anything else is the
        # model query itself.
        query = " ".join(part for part in (action.strip(), rest) if part).strip()

        if act in ("help", "帮助", "说明"):
            text = USAGE
        elif act in ("sub", "订阅"):
            text = await self._subscribe(umo, rest)
        elif act in ("unsub", "取消订阅", "退订"):
            text = await self._unsubscribe(umo, rest)
        elif act in ("list", "订阅列表"):
            text = self._render_subscriptions(umo)
        elif act in ("agents", "厂商", "厂商状态"):
            text = await self._render_agents_view()
        elif act in ("price", "prices", "价格", "价目表"):
            text = self._render_price_list(await self._fetch_prices())
        else:
            try:
                data = await self._fetch_status()
            except Exception as exc:
                logger.warning(f"[Mirasim] Status query failed: {exc!r}")
                text = f"获取 Mirasim 状态失败：{str(exc) or type(exc).__name__}"
            else:
                if query and query.lower() not in ALL_WORDS:
                    text = self._render_model(data, query, await self._fetch_prices())
                else:
                    show_all = bool(query)
                    if self.config.get("render_image", True):
                        path = await self._render_overview_image(data, show_all)
                        if path:
                            yield event.image_result(path)
                            return
                    text = self._render_overview(data, show_all)
        yield event.plain_result(text)

    def _render_subscriptions(self, umo: str) -> str:
        """Describe what the session subscribed to.

        Args:
            umo: Unified message origin of the session.

        Returns:
            The reply text.
        """
        wanted = self._subs.get(umo, [])
        if not wanted:
            return "本会话还没有订阅任何模型，发送 /mirasim sub <模型> 订阅。"
        if ALL_MODELS in wanted:
            return "本会话已订阅全部模型（包括之后新增的模型）。"
        return f"本会话订阅了 {len(wanted)} 项：\n" + "\n".join(
            f"· {entry}" for entry in wanted
        )

    async def _render_agents_view(self) -> str:
        """Render the per-vendor view built from each agent's summary block.

        Returns:
            The reply text.
        """
        try:
            data = await self._fetch_status()
        except Exception as exc:
            logger.warning(f"[Mirasim] Status query failed: {exc!r}")
            return f"获取 Mirasim 状态失败：{str(exc) or type(exc).__name__}"
        lines = ["Mirasim 厂商状态"]
        for record in _collect_agents(data).values():
            lines.append(f"· {record['name']}")
            for cohort_id, agent in record["entries"]:
                status, availability = _agent_status(agent, data)
                summary = agent.get("summary") or {}
                window = _fmt_window((summary.get("now") or {}).get("window"))
                p50 = _fmt_seconds((summary.get("latency") or {}).get("p50"))
                detail = _fmt_rate(availability, short=True)
                if p50 != "—":
                    detail += f" · p50 {p50}"
                if window:
                    detail += f" · {window}"
                reason = _reason_note(agent.get("reasons"))
                lines.append(
                    f"  {STATUS_ICONS.get(status, '❔')} "
                    f"{COHORT_NAMES.get(cohort_id, cohort_id)} · {detail}" + reason
                )
        _append_footer(lines, data)
        return "\n".join(lines)

    def _overview_groups(
        self, data: dict, show_all: bool
    ) -> tuple[list[dict], list[str], bool]:
        """Pick, filter and sort the models shown in the overview.

        Each ``display_models`` entry is either a plain model name (shows every
        cohort), a vendor name (shows every model of that vendor) or a
        ``model@cohort`` form (shows only that cohort). The forms may be
        mixed freely.

        Args:
            data: Status payload.
            show_all: Ignore the ``display_models`` setting.

        Returns:
            ``(groups, missing, filtered)`` where each group is one vendor with
            its models; problems sort first.
        """
        models = _collect_models(data)
        aliases = _model_aliases(models)
        configured = [
            str(item).strip()
            for item in self.config.get("display_models") or []
            if str(item).strip()
        ]
        filtered = bool(configured) and not show_all
        missing: list[str] = []
        # chosen maps model key -> set of cohort_ids, or None meaning every cohort.
        chosen: dict[str, set[str] | None] = {}
        if filtered:
            for raw in configured:
                model_part, cohort_part = _split_target(raw)
                if not model_part:
                    continue
                try:
                    keys = _match_keys(models, model_part, aliases)
                except LookupError as exc:
                    missing.append(raw)
                    logger.warning(f"[Mirasim] {exc}")
                    continue
                if not keys:
                    missing.append(raw)
                    continue
                # A cohort filter applies to the models that actually have it.
                picked = keys
                if cohort_part is not None:
                    picked = [
                        key
                        for key in keys
                        if cohort_part in {c for c, _m, _a in models[key]["entries"]}
                    ]
                    if not picked:
                        missing.append(raw)
                        continue
                for key in picked:
                    # Already chose this model with no cohort filter: nothing changes.
                    if key in chosen and chosen[key] is None:
                        continue
                    if cohort_part is None:
                        chosen[key] = None
                        continue
                    bucket = chosen.get(key)
                    if bucket is None and key not in chosen:
                        chosen[key] = {cohort_part}
                    elif bucket is not None:
                        bucket.add(cohort_part)
        else:
            chosen = dict.fromkeys(models)

        groups: dict[str, dict] = {}
        for key, cohort_filter in chosen.items():
            record = models[key]
            entries = record["entries"]
            if cohort_filter is not None:
                entries = [e for e in entries if e[0] in cohort_filter]
            if not entries:
                continue
            tagged = [
                (_model_status(model, data)[0], (cohort_id, model, agent))
                for cohort_id, model, agent in entries
            ]
            group = groups.setdefault(
                record["agent_id"] or record["agent"],
                {
                    "agent": record["agent"],
                    "agent_id": record["agent_id"],
                    "models": [],
                    "cohorts": [],
                    "counts": {},
                },
            )
            for cohort_id, _m, _a in entries:
                if cohort_id not in group["cohorts"]:
                    group["cohorts"].append(cohort_id)
            group["models"].append(
                {
                    "key": key,
                    "name": record["name"],
                    "entries": tagged,
                    "worst": _worst_status([s for s, _e in tagged]),
                }
            )

        order = {cid: i for i, cid in enumerate(COHORT_ORDER)}
        for group in groups.values():
            for status, _entry in (
                (s, e) for model in group["models"] for s, e in model["entries"]
            ):
                group["counts"][status] = group["counts"].get(status, 0) + 1
            group["cohorts"].sort(key=lambda c: order.get(c, len(COHORT_ORDER)))
            group["worst"] = _worst_status(
                [model["worst"] for model in group["models"]]
            )
            group["total"] = sum(len(m["entries"]) for m in group["models"])
            group["up"] = group["counts"].get("ok", 0)
        ordered = list(groups.values())
        if not filtered:
            # No display_models configured: vendors with problems first. Model
            # order inside a vendor stays as the payload lists it.
            ordered.sort(key=lambda g: (STATUS_ORDER.get(g["worst"], 99), g["agent"]))
        return ordered, missing, filtered

    def _render_overview(self, data: dict, show_all: bool) -> str:
        """Render the status list of the configured models, or of all of them.

        Args:
            data: Status payload.
            show_all: Ignore the ``display_models`` setting.

        Returns:
            The reply text.
        """
        groups, missing, filtered = self._overview_groups(data, show_all)
        up = sum(g["up"] for g in groups)
        total = sum(g["total"] for g in groups)
        lines = [f"Mirasim 模型状态 · 正常 {up}/{total}"]
        if not groups:
            lines.append("暂无模型数据。")
        shown_cohorts: set[str] = set()
        for group in groups:
            lines.append(f"{group['agent']} · {group['up']}/{group['total']} 正常")
            for model in group["models"]:
                segments = []
                for status, (cohort_id, m, _a) in model["entries"]:
                    shown_cohorts.add(cohort_id)
                    availability = _fmt_rate(_model_status(m, data)[1], short=True)
                    segments.append(
                        f"{STATUS_ICONS.get(status, '❔')}"
                        f"{COHORT_NAMES.get(cohort_id, cohort_id)} {availability}"
                    )
                lines.append(f"· {model['name']} — {' · '.join(segments)}")
        _append_footer(lines, data, filtered, cohorts=shown_cohorts)
        if missing:
            lines.append(f"未找到：{'、'.join(missing)}")
        return "\n".join(lines)

    async def _render_overview_image(self, data: dict, show_all: bool) -> str | None:
        """Render the overview as a card image through AstrBot's T2I service.

        Args:
            data: Status payload.
            show_all: Ignore the ``display_models`` setting.

        Returns:
            Path of the rendered image, or None when rendering failed and the
            caller should fall back to text.
        """
        groups, missing, filtered = self._overview_groups(data, show_all)
        counts: dict[str, int] = {}
        total_up = 0
        total_rows = 0
        for group in groups:
            for status, count in group["counts"].items():
                counts[status] = counts.get(status, 0) + count
            total_up += group["up"]
            total_rows += group["total"]

        cohort_states = _cohort_state(data)
        for group in groups:
            for model in group["models"]:
                cells = []
                for cohort_id in group["cohorts"]:
                    match = next(
                        (
                            (status, (cid, m, agent))
                            for status, (cid, m, agent) in model["entries"]
                            if cid == cohort_id
                        ),
                        None,
                    )
                    if match is None:
                        cells.append({"empty": True, "cohort": cohort_id})
                        continue
                    status, (cid, m, agent) = match
                    availability = _model_status(m, data)[1]
                    p50 = _fmt_seconds((m.get("latency") or {}).get("p50"))
                    metrics = _fmt_rate(availability, short=True)
                    if p50 != "—":
                        metrics += f" · {p50}"
                    note = _reason_note(agent.get("reasons"))
                    cells.append(
                        {
                            "empty": False,
                            "cohort": cid,
                            "cohort_label": COHORT_NAMES.get(cid, cid),
                            "status": status,
                            "label": STATUS_NAMES.get(status, status),
                            "metrics": metrics,
                            "note": note,
                            "history": _history_bars(m.get("cells"), data),
                        }
                    )
                model["cells"] = cells

        interval = data.get("cellSeconds")
        tmpl_data = {
            "groups": groups,
            "counts": counts,
            "total_up": total_up,
            "total_rows": total_rows,
            "cohort_states": cohort_states,
            "cohort_names": COHORT_NAMES,
            "status_names": STATUS_NAMES,
            "missing": missing,
            "filtered": filtered,
            "checked_at": _fmt_time(data.get("generatedAt")),
            "through_at": _fmt_time(data.get("dataThrough")),
            "interval": (
                _fmt_duration(interval)
                if isinstance(interval, (int, float)) and interval > 0
                else ""
            ),
            "notes": [str(n) for n in (data.get("notes") or []) if str(n).strip()],
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

    def _render_model(self, data: dict, query: str, prices: pricing.PriceList) -> str:
        """Render the detail view of one model, or one of its cohorts.

        Args:
            data: Status payload.
            query: The user's ``model[@cohort]`` input.
            prices: The price list to look the model up in.

        Returns:
            The reply text.
        """
        record, cohort_part, error = self._resolve_model_and_entry(data, query)
        if error:
            return error
        title = record["name"] + (f"@{cohort_part}" if cohort_part else "")
        agent = record["agent"]
        lines = [f"· {title}" + (f"（{agent}）" if agent else "")]

        for cohort_id, model, agent_blob in record["entries"]:
            label = COHORT_NAMES.get(cohort_id, cohort_id)
            status, availability = _model_status(model, data)
            icon = STATUS_ICONS.get(status, "❔")
            name = STATUS_NAMES.get(status, status)
            window = _fmt_window((model.get("now") or {}).get("window"))
            latency = model.get("latency") or {}
            availability_stats = model.get("availability") or {}

            header = f"{icon} {label}池 {name}"
            # One parenthetical holding both the failure split and the current
            # availability; two adjacent "（）" read as a typo.
            extras = []
            reason_text = _format_reasons(agent_blob.get("reasons"))
            if reason_text:
                extras.append(reason_text)
            if status in ("down", "degraded") and isinstance(
                availability, (int, float)
            ):
                rate = f"可用率 {_fmt_rate(availability, short=True)}"
                if window:
                    rate += f" · 最近 {window}"
                extras.append(rate)
            if extras:
                header += f"（{' · '.join(extras)}）"
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
            same_model = model.get("sameModel")
            if isinstance(same_model, (int, float)) and same_model < 100:
                lines.append(f"  模型不匹配 {100 - same_model:.1f}%")
            intel = model.get("intel")
            if isinstance(intel, dict):
                swapped = intel.get("swapped")
                cut = intel.get("cut")
                mismatched = intel.get("mismatched")
                if isinstance(swapped, (int, float)) and swapped:
                    lines.append(f"  路由切换 {swapped:.1f}%")
                if isinstance(mismatched, (int, float)) and mismatched:
                    lines.append(f"  响应不符 {mismatched:.1f}%")
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
                            f"  近 24h 曾异常（{_fmt_time(went_down.isoformat())}）"
                        )
                stripe = "".join(STRIPE_ICONS.get(b, "⬜") for b in bars)
                lines.append(
                    f"  近 24h（旧→新，每格 {_cell_width(data)}）：\n  {stripe}"
                )
        _append_footer(lines, data, cohorts={c for c, _m, _a in record["entries"]})
        price = prices.find(record["agent"], record["name"])
        if price is not None:
            lines.append("")
            lines.append("💲 价格（每百万 token 扣的额度，$1 = 1 额度）")
            lines.append(f"· {_price_bits(price)}")
            if price.discounted:
                lines.append(
                    f"· ⏳{price.badge or price.tag}，优惠随时可能结束，结束后按刊例价扣"
                )
        return "\n".join(lines)

    def _render_price_list(self, prices: pricing.PriceList) -> str:
        """Render the whole price list, grouped by vendor.

        Args:
            prices: The price list.

        Returns:
            The reply text.
        """
        if not prices.entries:
            return "暂时取不到 Mirasim 的价目表，稍后再试。"
        lines = [
            "Mirasim 模型价目表",
            "每百万 token 扣的额度，$1 = 1 额度",
            "",
        ]
        vendor = None
        for entry in prices.entries.values():
            if entry.vendor != vendor:
                vendor = entry.vendor
                badge = f" ⏳{entry.badge}" if entry.badge else ""
                lines.append(f"{vendor}{badge}")
            lines.append(f"· {entry.name} — {_price_bits(entry)}")
        lines.append("")
        lines.append("· 缓存写入也按官方 API 价扣；用自己的账号或 API 密钥不扣额度")
        if prices.updated:
            lines.append(f"· 更新于 {prices.updated}")
        return "\n".join(lines)

    def _resolve_model_and_entry(
        self, data: dict, target: str
    ) -> tuple[dict | None, str | None, str]:
        """Resolve user input of ``model[@cohort]`` against the payload.

        Args:
            data: Status payload.
            target: Raw user input, e.g. ``opus 5.5@paid``.

        Returns:
            ``(record, cohort_part, "")`` on hit, ``(None, None, error)`` on miss.
        """
        model_part, cohort_part = _split_target(target)
        models = _collect_models(data)
        if not model_part:
            return None, None, "用法：/mirasim <模型>[@<free|paid|cloud>]"
        try:
            key = _resolve_key(
                list(models),
                model_part,
                "发送 /mirasim all 查看全部模型。",
                _model_aliases(models),
            )
        except LookupError as exc:
            return None, None, str(exc)
        record = dict(models[key])
        entries = record["entries"]
        if cohort_part is None:
            return record, None, ""
        matched = [e for e in entries if e[0].lower() == cohort_part]
        if not matched:
            return (
                None,
                None,
                (
                    f"未知池「{cohort_part}」，{record['name']} 可用："
                    f"{'、'.join(f'{c}（{COHORT_NAMES.get(c, c)}）' for c, _m, _a in entries)}"
                ),
            )
        record["entries"] = matched
        return record, matched[0][0], ""

    async def _subscribe(self, umo: str, target: str) -> str:
        """Subscribe a session to a model, one of its cohorts, or all models.

        Args:
            umo: Unified message origin of the session.
            target: A model name, ``model@cohort``, or one of ``ALL_WORDS``.

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
        record, cohort_part, error = self._resolve_model_and_entry(data, target)
        if error:
            return error
        entry = f"{record['key']}@{cohort_part}" if cohort_part else record["key"]
        if entry in wanted:
            return f"本会话已订阅 {record['name']}{f'@{cohort_part}' if cohort_part else ''}。"

        # Check whether an existing subscription already covers this target.
        sample_keys = [f"{record['key']}@{c}" for c, _m, _a in record["entries"]]
        aliases = _model_aliases(_collect_models(data))
        if any(_covers(e, k, aliases) for e in wanted for k in sample_keys):
            return "本会话的现有订阅已覆盖该目标。"

        self._subs[umo] = [*wanted, entry]
        await self.put_kv_data("subscriptions", self._subs)

        if cohort_part:
            cohort_id, model, _a = record["entries"][0]
            status, _ = _model_status(model, data)
            return (
                f"已订阅 {record['name']}@{cohort_id}"
                f"（{COHORT_NAMES.get(cohort_id, cohort_id)}池，"
                f"当前 {STATUS_ICONS.get(status, '❔')} {STATUS_NAMES.get(status, status)}），"
                "状态变化时会推送到本会话。"
            )
        statuses = [_model_status(m, data)[0] for _c, m, _a in record["entries"]]
        worst = _worst_status(statuses)
        return (
            f"已订阅 {record['name']} 的全部 {len(record['entries'])} 个池"
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
        needle = _slug(model_part)
        # The stored entry may still carry the old vendor-prefixed name, so the
        # aliases decide whether it names the same model as the input.
        try:
            aliases = _model_aliases(_collect_models(await self._fetch_status()))
        except Exception:
            aliases = {}
        matched = [
            e
            for e in wanted
            if (
                _slug(_split_target(e)[0]) == needle
                or aliases.get(_slug(_split_target(e)[0])) == needle
            )
            and (cohort_part is None or _split_target(e)[1] == cohort_part)
        ]
        if not matched:
            hint = "发送 /mirasim list 查看本会话的订阅。"
            return f"本会话没有订阅「{target}」，{hint}"

        remaining = [e for e in wanted if e not in matched]
        if remaining:
            self._subs[umo] = remaining
        else:
            self._subs.pop(umo, None)
        await self.put_kv_data("subscriptions", self._subs)
        return f"已取消订阅 {'、'.join(matched)}。"


def _cell_width(data: dict) -> str:
    """Describe one history cell's width for the stripe caption.

    Args:
        data: Status payload.

    Returns:
        Text such as ``30分钟``; ``一格`` when the payload has no interval.
    """
    seconds = data.get("cellSeconds")
    if isinstance(seconds, (int, float)) and seconds > 0:
        return _fmt_duration(seconds)
    return "一格"


def _append_footer(
    lines: list[str],
    data: dict,
    filtered: bool = False,
    cohorts: object = None,
) -> None:
    """Append the timestamp / interval footer to a text reply.

    Args:
        lines: Reply lines being assembled, modified in place.
        data: Status payload.
        filtered: True when only configured models are shown.
        cohorts: Cohort IDs to report pool states for; None means all of them.
    """
    checked_at = _fmt_time(data.get("generatedAt"))
    footer = f"检测于 {checked_at}"
    through = _fmt_time(data.get("dataThrough"))
    if through != "—" and through != checked_at:
        footer += f" · 数据截至 {through}"
    seconds = data.get("cellSeconds")
    if isinstance(seconds, (int, float)) and seconds > 0:
        footer += f" · 每 {_fmt_duration(seconds)}一格"
    lines.append(footer)
    states = _cohort_state(data)
    if isinstance(cohorts, (set, list, tuple)):
        states = {c: s for c, s in states.items() if c in cohorts}
    if states:
        lines.append(
            "池状态："
            + " · ".join(
                f"{COHORT_NAMES.get(cid, cid)} {STATUS_NAMES.get(state, state)}"
                for cid, state in states.items()
            )
        )
    for note in data.get("notes") or []:
        text = str(note).strip()
        if text:
            lines.append(f"注：{text}")
    if filtered:
        lines.append("仅展示配置的模型，发送 /mirasim all 查看全部。")


def _change_line(
    new_status: str,
    model: dict,
    cohort_id: str,
    reason: str,
    data: dict,
    models: dict[str, dict],
) -> str:
    """Build one push line describing a confirmed transition.

    Args:
        new_status: Status the model just settled into.
        model: Model dict from the current payload.
        cohort_id: Cohort the model belongs to.
        reason: Bare agent failure breakdown text (``限流79%·故障21%``), or empty.
        data: Whole payload, for cell timing.
        models: Model index of the current payload, for the display name.

    Returns:
        One line of push text.
    """
    key = _slug(model.get("name")) or _slug(model.get("id"))
    name = (models.get(key) or {}).get("name") or key
    label = COHORT_NAMES.get(cohort_id, cohort_id)
    icon = CHANGE_ICONS.get(new_status, "⚪")
    _status, availability = _model_status(model, data)
    availability_text = _fmt_rate(availability, short=True)
    bars = _history_bars(model.get("cells"), data)
    cell_seconds = data.get("cellSeconds")

    tail = f" · {reason}" if reason else ""

    if new_status == "down":
        line = f"{icon} {name}（{label}）异常（可用率 {availability_text}"
        streak = _last_streak(bars, ("down",))
        if streak and isinstance(cell_seconds, (int, float)):
            since = _cell_time(data, streak[0])
            if since is not None:
                line += f"，{_fmt_time(since.isoformat())} 起"
        return line + "）" + tail

    if new_status == "degraded":
        return f"{icon} {name}（{label}）降级（可用率 {availability_text}）" + tail

    if new_status == "ok":
        p50 = _fmt_seconds((model.get("latency") or {}).get("p50"))
        line = f"{icon} {name}（{label}）已恢复，p50 {p50}，可用率 {availability_text}"
        streak = _last_streak(bars, ("down", "degraded"))
        if streak and isinstance(cell_seconds, (int, float)):
            outage_secs = streak[1] * cell_seconds
            if outage_secs > 0:
                prefix = "超过" if streak[0] == 0 else "约"
                line += f"，异常持续{prefix} {_fmt_duration(outage_secs)}"
        return line + tail

    return f"{icon} {name}（{label}）{STATUS_NAMES.get(new_status, new_status)}"
