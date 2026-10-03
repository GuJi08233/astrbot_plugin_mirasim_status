"""Tests for change detection, pushes and commands of the Mirasim plugin.

Run from the AstrBot root so the plugin imports by its package path. Point
ASTRBOT_ROOT elsewhere so importing astrbot.core leaves the real data/ alone:
    ASTRBOT_ROOT=/tmp/mirasim_test uv run python -m unittest \
        data.plugins.astrbot_plugin_mirasim_status.tests.test_main
"""

import asyncio
import importlib
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

main_mod = importlib.import_module("data.plugins.astrbot_plugin_mirasim_status.main")
MirasimStatus = main_mod.MirasimStatus

BASE = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def iso(minutes: float) -> str:
    """Return BASE + minutes as the monitor's ISO format."""
    moment = BASE + timedelta(minutes=minutes)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def model(model_id: str, statuses: str, active: bool = True, **extra) -> dict:
    """Build a model entry whose history is one sample per 5 minutes.

    Args:
        model_id: The model ID.
        statuses: History from old to new, ``U`` for up and ``d`` for down.
        active: Whether the model is in the current inventory.
        **extra: Fields overriding the derived ones.
    """
    history = [
        {"at": iso(i * 5), "status": "up" if c == "U" else "down"}
        for i, c in enumerate(statuses)
    ]
    last = history[-1] if history else None
    entry = {
        "id": model_id,
        "active": active,
        "status": last["status"] if last else "unknown",
        "last_checked_at": last["at"] if last else None,
        "last_success_at": None,
        "last_failure_at": None,
        "current_stable_since": None,
        "stable_seconds": 0,
        "latency_ms": 2345.6,
        "last_error": None if not last or last["status"] == "up" else "HTTP 503",
        "samples_24h": len(history),
        "success_rate_24h": 100.0 * statuses.count("U") / len(statuses)
        if statuses
        else 0,
        "avg_latency_ms_24h": 2000.0,
        "history": history,
    }
    entry.update(extra)
    return entry


def payload(*models: dict, now_minutes: float | None = None) -> dict:
    """Wrap models into a status document checked right after the last sample."""
    newest = max((len(m["history"]) for m in models), default=1)
    now = iso(now_minutes if now_minutes is not None else newest * 5)
    return {
        "models": list(models),
        "summary": {},
        "inventory": {"error": None},
        "now": now,
        "interval_seconds": 300,
        "stale_after_seconds": 690,
        "scan": {"last_finished_at": now},
    }


def make_plugin(**config) -> MirasimStatus:
    context = MagicMock()
    context.send_message = AsyncMock(return_value=True)
    plugin = MirasimStatus(context, {"confirm_samples": 2, **config})
    plugin.put_kv_data = AsyncMock()
    plugin.get_kv_data = AsyncMock(side_effect=lambda key, default: default)
    return plugin


def run(coro):
    return asyncio.run(coro)


def sent_messages(plugin: MirasimStatus) -> dict[str, str]:
    """Map each pushed session to the text it received."""
    result = {}
    for call in plugin.context.send_message.await_args_list:
        umo, chain = call.args
        result[umo] = chain.chain[0].text
    return result


async def command(plugin: MirasimStatus, *args: str, umo: str = "qq:GroupMessage:1"):
    event = MagicMock()
    event.unified_msg_origin = umo
    event.plain_result = lambda text: text
    replies = [reply async for reply in plugin.cmd_mirasim(event, *args)]
    return replies[0]


class ResolveIdTest(unittest.TestCase):
    IDS = ["claude-opus-5", "claude-opus-5-5", "claude-sonnet-5-5", "kimi-k3"]

    def test_exact_match_beats_longer_ids_containing_it(self):
        self.assertEqual(
            main_mod._resolve_id(self.IDS, "claude-opus-5", ""), "claude-opus-5"
        )

    def test_unique_part_and_case_are_forgiven(self):
        self.assertEqual(main_mod._resolve_id(self.IDS, "KIMI", ""), "kimi-k3")
        self.assertEqual(
            main_mod._resolve_id(self.IDS, "opus-5-5", ""), "claude-opus-5-5"
        )

    def test_ambiguous_input_lists_the_candidates(self):
        with self.assertRaises(LookupError) as ctx:
            main_mod._resolve_id(self.IDS, "5-5", "")
        self.assertIn("claude-opus-5-5、claude-sonnet-5-5", str(ctx.exception))

    def test_unknown_input_carries_the_hint(self):
        with self.assertRaises(LookupError) as ctx:
            main_mod._resolve_id(self.IDS, "gemini", "试试 list")
        self.assertIn("试试 list", str(ctx.exception))


class EffectiveStatusTest(unittest.TestCase):
    def test_result_older_than_the_limit_is_stale(self):
        entry = model("m", "UU")
        self.assertEqual(main_mod._effective_status(entry, payload(entry)), "up")
        late = payload(entry, now_minutes=5 + 690 / 60 + 1)
        self.assertEqual(main_mod._effective_status(entry, late), "stale")

    def test_never_checked_is_unknown(self):
        entry = model("m", "")
        self.assertEqual(main_mod._effective_status(entry, payload(entry)), "unknown")


class ChangeDetectionTest(unittest.TestCase):
    def test_first_poll_records_a_silent_baseline(self):
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        run(plugin._check_changes(payload(model("a", "UU"), model("b", "dd"))))
        self.assertEqual(plugin._states, {"a": "up", "b": "down"})
        plugin.put_kv_data.assert_awaited_once_with("states", plugin._states)
        plugin.context.send_message.assert_not_awaited()

    def test_single_failed_sample_is_not_pushed(self):
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {"a": "up"}
        run(plugin._check_changes(payload(model("a", "UUUd"))))
        self.assertEqual(plugin._states, {"a": "up"})
        plugin.context.send_message.assert_not_awaited()

    def test_confirm_samples_of_one_pushes_every_flip(self):
        plugin = make_plugin(confirm_samples=1)
        plugin._subs = {"s": ["*"]}
        plugin._states = {"a": "up"}
        run(plugin._check_changes(payload(model("a", "UUUd"))))
        self.assertIn("🔴 a 故障", sent_messages(plugin)["s"])

    def test_confirmed_failure_reaches_only_its_subscribers_in_one_message(self):
        plugin = make_plugin()
        plugin._states = {"a": "up", "b": "up", "c": "up"}
        plugin._subs = {
            "group-all": ["*"],
            "group-a": ["a"],
            "group-c": ["c"],
        }
        data = payload(model("a", "UUdd"), model("b", "Udd"), model("c", "UUU"))
        run(plugin._check_changes(data))

        sent = sent_messages(plugin)
        self.assertEqual(set(sent), {"group-all", "group-a"})
        self.assertIn("🔴 a 故障：上游 HTTP 503", sent["group-all"])
        self.assertIn("🔴 b 故障", sent["group-all"])
        self.assertTrue(sent["group-all"].startswith("Mirasim 状态变化\n"))
        self.assertNotIn("b 故障", sent["group-a"])
        self.assertEqual(plugin._states, {"a": "down", "b": "down", "c": "up"})

    def test_recovery_reports_the_outage_length(self):
        plugin = make_plugin()
        plugin._subs = {"s": ["a"]}
        plugin._states = {"a": "down"}
        # Failures at minutes 5 and 10, back up at 15: a 10-minute outage.
        run(plugin._check_changes(payload(model("a", "UddUU"))))
        text = sent_messages(plugin)["s"]
        self.assertIn("🟢 a 已恢复，当前耗时 2.3s", text)
        self.assertIn("故障持续约 10分钟", text)

    def test_outage_older_than_the_history_is_marked_as_a_lower_bound(self):
        plugin = make_plugin()
        plugin._subs = {"s": ["a"]}
        plugin._states = {"a": "down"}
        run(plugin._check_changes(payload(model("a", "ddUU"))))
        self.assertIn("故障持续超过 10分钟", sent_messages(plugin)["s"])

    def test_models_outside_the_inventory_are_ignored(self):
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {"gone": "up"}
        run(plugin._check_changes(payload(model("gone", "Udd", active=False))))
        self.assertEqual(plugin._states, {"gone": "up"})
        plugin.context.send_message.assert_not_awaited()

    def test_failed_push_does_not_stop_the_other_sessions(self):
        plugin = make_plugin()
        plugin._subs = {"broken": ["*"], "fine": ["*"]}
        plugin._states = {"a": "up"}
        plugin.context.send_message.side_effect = [RuntimeError("boom"), True]
        run(plugin._check_changes(payload(model("a", "Udd"))))
        self.assertEqual(plugin.context.send_message.await_count, 2)
        self.assertEqual(plugin._states, {"a": "down"})


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.data = payload(
            model("claude-opus-5-5", "UUU", stable_seconds=600),
            model("claude-sonnet-5-5", "UUU"),
            model("kimi-k3", "Udd", last_error="upstream_truncated"),
            model("old-model", "UU", active=False),
        )
        self.plugin = make_plugin()
        self.plugin._fetch_status = AsyncMock(return_value=self.data)

    def test_overview_lists_problems_first_and_skips_retired_models(self):
        text = run(command(self.plugin))
        lines = text.splitlines()
        self.assertEqual(lines[0], "Mirasim 模型状态 · 可用 2/3")
        self.assertEqual(lines[1], "❌ kimi-k3 · 上游输出被截断")
        self.assertIn("✅ claude-opus-5-5 · 2.3s · 24h 100%", lines)
        self.assertNotIn("old-model", text)
        self.assertIn("每 5分钟一轮", text)

    def test_configured_models_narrow_the_overview_until_all_is_asked(self):
        self.plugin.config["display_models"] = ["claude-opus-5-5", "gpt-9"]
        text = run(command(self.plugin))
        self.assertIn("可用 1/1", text)
        self.assertNotIn("kimi-k3", text)
        self.assertIn("未找到：gpt-9", text)
        self.assertIn("/mirasim all", text)

        everything = run(command(self.plugin, "all"))
        self.assertIn("kimi-k3", everything)
        self.assertNotIn("未找到", everything)

    def test_detail_view_shows_reason_and_history(self):
        text = run(command(self.plugin, "kimi"))
        self.assertTrue(text.startswith("❌ kimi-k3 · 失败"))
        self.assertIn("失败原因：上游输出被截断", text)
        self.assertIn("最近 3 次采样（旧→新）：\n🟩🟥🟥", text)

    def test_retired_models_stay_queryable(self):
        text = run(command(self.plugin, "old-model"))
        self.assertIn("已退出监控目录", text)

    def test_fetch_failure_is_reported(self):
        self.plugin._fetch_status = AsyncMock(side_effect=TimeoutError())
        self.assertEqual(
            run(command(self.plugin)), "获取 Mirasim 状态失败：TimeoutError"
        )

    def test_subscription_lifecycle(self):
        plugin = self.plugin
        reply = run(command(plugin, "sub", "opus"))
        self.assertIn("已订阅 claude-opus-5-5（当前 ✅ 可用）", reply)
        self.assertEqual(plugin._subs, {"qq:GroupMessage:1": ["claude-opus-5-5"]})
        plugin.put_kv_data.assert_awaited_with("subscriptions", plugin._subs)

        self.assertIn("匹配到多个模型", run(command(plugin, "sub", "5-5")))
        self.assertIn("已订阅", run(command(plugin, "sub", "claude-opus-5-5")))
        run(command(plugin, "sub", "kimi"))
        listing = run(command(plugin, "list"))
        self.assertIn("订阅了 2 个模型", listing)
        self.assertIn("· kimi-k3", listing)

        # Another session keeps its own list.
        self.assertIn("还没有订阅", run(command(plugin, "list", umo="qq:Friend:9")))

        self.assertEqual(run(command(plugin, "unsub", "kimi")), "已取消订阅 kimi-k3。")
        run(command(plugin, "unsub", "opus"))
        self.assertEqual(plugin._subs, {})

    def test_subscribing_to_everything(self):
        plugin = self.plugin
        run(command(plugin, "sub", "claude-opus-5-5"))
        self.assertIn("已订阅全部模型", run(command(plugin, "sub", "all")))
        self.assertEqual(plugin._subs, {"qq:GroupMessage:1": ["*"]})
        self.assertIn("无需单独订阅", run(command(plugin, "sub", "kimi")))
        self.assertIn("unsub all", run(command(plugin, "unsub", "kimi")))
        self.assertEqual(
            run(command(plugin, "unsub", "all")), "已取消本会话的全部订阅。"
        )
        self.assertEqual(plugin._subs, {})

    def test_cannot_subscribe_to_a_retired_model(self):
        reply = run(command(self.plugin, "sub", "old-model"))
        self.assertIn("没有找到", reply)
        self.assertEqual(self.plugin._subs, {})


if __name__ == "__main__":
    unittest.main()
