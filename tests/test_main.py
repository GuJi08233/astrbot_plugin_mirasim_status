"""Tests for change detection, pushes and commands of the Mirasim plugin.

Run from the AstrBot root so the plugin imports by its package path. Point
ASTRBOT_ROOT elsewhere so importing astrbot.core leaves the real data/ alone:
    ASTRBOT_ROOT=/tmp/mirasim_test uv run python -m unittest \
        data.plugins.astrbot_plugin_mirasim_status.tests.test_main
"""

import asyncio
import importlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import jinja2

from astrbot.core.utils.t2i.network_strategy import inject_shiki_runtime

main_mod = importlib.import_module("data.plugins.astrbot_plugin_mirasim_status.main")
MirasimStatus = main_mod.MirasimStatus

BASE = datetime(2026, 10, 5, 8, 0, tzinfo=timezone.utc)


def iso(hours: float) -> str:
    """Return BASE + hours as the monitor's ISO format."""
    moment = BASE + timedelta(hours=hours)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def make_model(
    model_id: str,
    name: str,
    *,
    status: str = "ok",
    availability: float | None = 100.0,
    p50: float | None = 5.0,
    h24: float = 99.0,
    d7: float = 99.0,
    cells: list[int] | None = None,
    same_model: int = 100,
    intel: dict | None = None,
) -> dict:
    """Build one model entry mirasim.ai-style (a value-per-cell array)."""
    if cells is None:
        cells = [1000] * 48
    return {
        "id": model_id,
        "name": name,
        "now": {"status": status, "availability": availability, "window": "15m"},
        "availability": {"h24": h24, "d7": d7},
        "latency": {"p50": p50, "p95": (p50 * 6 if p50 is not None else None)},
        "sameModel": same_model,
        "intel": intel,
        "cells": cells,
        "merged": [],
    }


def make_agent(
    agent_id: str,
    name: str,
    models: list[dict],
    *,
    reasons: list[dict] | None = None,
) -> dict:
    """Wrap models under an agent blob; agent summary mirrors the first model."""
    summary = dict(models[0]) if models else {}
    summary.pop("id", None)
    summary.pop("name", None)
    return {
        "id": agent_id,
        "name": name,
        "summary": summary,
        "models": models,
        "reasons": reasons,
    }


def make_payload(
    *,
    agents_free: list[dict] | None = None,
    agents_paid: list[dict] | None = None,
    agents_cloud: list[dict] | None = None,
    generated_h: float = 24.0,
) -> dict:
    """Build the full status payload from per-cohort agent lists."""
    return {
        "schema": 2,
        "generatedAt": iso(generated_h),
        "dataThrough": iso(generated_h),
        "cellSeconds": 1800,
        "cellsStart": iso(0),
        "thresholds": {"good": 99, "warn": 95, "minTurns": 20},
        "notes": [],
        "cohorts": [
            {"id": "free", "state": "ok", "agents": agents_free or []},
            {"id": "paid", "state": "ok", "agents": agents_paid or []},
            {"id": "cloud", "state": "ok", "agents": agents_cloud or []},
        ],
    }


def make_plugin(**config) -> MirasimStatus:
    context = MagicMock()
    context.send_message = AsyncMock(return_value=True)
    # Text replies by default, so no test reaches the real T2I service.
    plugin = MirasimStatus(
        context, {"confirm_samples": 2, "render_image": False, **config}
    )
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
    event.image_result = lambda path: ("image", path)
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


class CellStatusTest(unittest.TestCase):
    """The plugin uses 90/50 thresholds, much looser than the site's 99/95."""

    def test_per_threshold_bucket(self):
        # 90% availability (900/1000) and above counts as 正常.
        self.assertEqual(main_mod._cell_status(1000, 90, 50), "ok")
        self.assertEqual(main_mod._cell_status(900, 90, 50), "ok")
        # 50–89.9% is 不稳定.
        self.assertEqual(main_mod._cell_status(899, 90, 50), "warn")
        self.assertEqual(main_mod._cell_status(500, 90, 50), "warn")
        # Below 50% is 中断.
        self.assertEqual(main_mod._cell_status(499, 90, 50), "down")
        self.assertEqual(main_mod._cell_status(0, 90, 50), "down")

    def test_gap_is_unknown(self):
        self.assertEqual(main_mod._cell_status(-1, 90, 50), "unknown")
        self.assertEqual(main_mod._cell_status(None, 90, 50), "unknown")

    def test_history_bars_use_loose_plugin_thresholds(self):
        """A model at 95% availability shows as 正常 despite the site warning."""
        cells = [950] * 48  # 95% everywhere — site says warn, plugin says ok.
        data = make_payload(
            agents_free=[make_agent("a", "A", [make_model("m", "m", cells=cells)])],
        )
        bars = main_mod._history_bars(cells, data)
        self.assertEqual(bars, ["ok"] * 48)


class NowStatusTest(unittest.TestCase):
    """The plugin derives status from availability, not from now.status."""

    def _model(self, status: str, availability: float) -> dict:
        return make_model("m", "m", status=status, availability=availability)

    def test_site_warn_but_plugin_ok(self):
        # Site would say warn at 97%, plugin says ok (97 >= 90).
        model = self._model("warn", 97.0)
        self.assertEqual(main_mod._now_status_of(model), ("ok", 97.0))

    def test_site_down_but_plugin_warn(self):
        # Site would say down at 91.9%, plugin says warn (50 ≤ 91.9 < 90 is False, so warn).
        model = self._model("down", 70.0)
        self.assertEqual(main_mod._now_status_of(model), ("warn", 70.0))

    def test_really_down(self):
        model = self._model("down", 30.0)
        self.assertEqual(main_mod._now_status_of(model), ("down", 30.0))

    def test_no_data(self):
        model = self._model("down", None)
        # availability=None → nodata
        model["now"]["availability"] = None
        self.assertEqual(main_mod._now_status_of(model)[0], "nodata")


class ChangeDetectionTest(unittest.TestCase):
    """The plugin watches now.status across two consecutive polls."""

    def _payload_with(self, model_states: dict[str, str]) -> dict:
        """One Claude model across all three pools in the given states."""
        return make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "claude-opus-5-5",
                            "opus 5.5",
                            status=model_states["free"],
                            availability=0.0
                            if model_states["free"] == "down"
                            else 100.0,
                        )
                    ],
                )
            ],
            agents_paid=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "claude-opus-5-5",
                            "opus 5.5",
                            status=model_states["paid"],
                            availability=0.0
                            if model_states["paid"] == "down"
                            else 100.0,
                        )
                    ],
                )
            ],
            agents_cloud=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "claude-opus-5-5",
                            "opus 5.5",
                            status=model_states["cloud"],
                            availability=0.0
                            if model_states["cloud"] == "down"
                            else 100.0,
                        )
                    ],
                )
            ],
        )

    def test_first_poll_records_a_silent_baseline(self):
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        data = self._payload_with({"free": "down", "paid": "down", "cloud": "ok"})
        run(plugin._check_changes(data))
        self.assertEqual(
            plugin._states,
            {
                "claude-opus-5-5@free": "down",
                "claude-opus-5-5@paid": "down",
                "claude-opus-5-5@cloud": "ok",
            },
        )
        plugin.put_kv_data.assert_awaited_once_with("states", plugin._states)
        plugin.context.send_message.assert_not_awaited()

    def test_single_poll_flip_does_not_push_with_confirm_two(self):
        """A change lasting one poll is just a blip."""
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {
            "claude-opus-5-5@free": "ok",
            "claude-opus-5-5@paid": "ok",
            "claude-opus-5-5@cloud": "ok",
        }
        prev = self._payload_with({"free": "ok", "paid": "ok", "cloud": "ok"})
        cur = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        run(plugin._check_changes(cur, prev))
        # prev was ok so the change hasn't been seen twice in a row.
        plugin.context.send_message.assert_not_awaited()

    def test_confirmed_failure_pushes_per_cohort(self):
        """A failure seen on two consecutive polls fires."""
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {
            "claude-opus-5-5@free": "ok",
            "claude-opus-5-5@paid": "ok",
            "claude-opus-5-5@cloud": "ok",
        }
        prev = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        cur = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        run(plugin._check_changes(cur, prev))
        sent = sent_messages(plugin)
        self.assertIn("🔴 claude-opus-5-5（体验）异常", sent["s"])
        self.assertNotIn("claude-opus-5-5（订阅）异常", sent["s"])
        self.assertEqual(plugin._states["claude-opus-5-5@free"], "down")

    def test_only_subscribed_cohort_gets_the_push(self):
        plugin = make_plugin()
        plugin._subs = {
            "s-paid": ["claude-opus-5-5@paid"],
            "s-free": ["claude-opus-5-5@free"],
            "s-all-cohorts": ["claude-opus-5-5"],
        }
        plugin._states = {
            "claude-opus-5-5@free": "ok",
            "claude-opus-5-5@paid": "ok",
            "claude-opus-5-5@cloud": "ok",
        }
        prev = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        cur = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        run(plugin._check_changes(cur, prev))
        sent = sent_messages(plugin)
        self.assertIn("s-free", sent)
        self.assertIn("s-all-cohorts", sent)
        self.assertNotIn("s-paid", sent)

    def test_confirm_one_pushes_immediately(self):
        plugin = make_plugin(confirm_samples=1)
        plugin._subs = {"s": ["*"]}
        plugin._states = {
            "claude-opus-5-5@free": "ok",
            "claude-opus-5-5@paid": "ok",
            "claude-opus-5-5@cloud": "ok",
        }
        cur = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        run(plugin._check_changes(cur, None))
        sent = sent_messages(plugin)
        self.assertIn("🔴 claude-opus-5-5（体验）异常", sent["s"])

    def test_recovery_lists_outage_duration(self):
        """The ok push reports how long down was visible inside the 24h cells."""
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {
            "claude-opus-5-5@free": "down",
        }
        # 41 ok cells, then 6 down cells (3 hours), then 1 ok cell.
        cells_with_outage = [1000] * 41 + [0] * 6 + [1000]
        # prev already shows the recovery (status ok), so two polls of "ok"
        # back-to-back confirm the transition.
        prev = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "claude-opus-5-5",
                            "opus 5.5",
                            status="ok",
                            availability=100.0,
                            p50=4.2,
                            cells=cells_with_outage,
                        )
                    ],
                )
            ],
            agents_paid=[],
            agents_cloud=[],
        )
        cur = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "claude-opus-5-5",
                            "opus 5.5",
                            status="ok",
                            availability=100.0,
                            p50=4.2,
                            cells=cells_with_outage,
                        )
                    ],
                )
            ],
            agents_paid=[],
            agents_cloud=[],
        )
        run(plugin._check_changes(cur, prev))
        sent = sent_messages(plugin)
        self.assertIn("🟢 claude-opus-5-5（体验）已恢复", sent["s"])
        self.assertIn("p50 4.2s", sent["s"])
        # Cells show a 6-cell outage (3 hours).
        self.assertIn("异常持续约 3小时", sent["s"])


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.data = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "claude-opus-5-5",
                            "opus 5.5",
                            status="down",
                            availability=0.0,
                            p50=None,
                        ),
                        make_model("claude-fable-5-1", "fable 5.1"),
                    ],
                    reasons=[
                        {"class": "throttle", "share": 78.5},
                        {"class": "outage", "share": 21.4},
                    ],
                ),
                make_agent(
                    "kimi-code",
                    "Kimi",
                    [make_model("kimi-k3", "k3")],
                ),
            ],
            agents_paid=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "claude-opus-5-5",
                            "opus 5.5",
                            status="down",
                            availability=0.0,
                            p50=None,
                        ),
                        make_model("claude-fable-5-1", "fable 5.1"),
                    ],
                ),
                make_agent(
                    "kimi-code",
                    "Kimi",
                    [make_model("kimi-k3", "k3")],
                ),
            ],
            agents_cloud=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model("claude-opus-5-5", "opus 5.5"),
                        make_model("claude-fable-5-1", "fable 5.1"),
                    ],
                ),
                make_agent(
                    "kimi-code",
                    "Kimi",
                    [make_model("kimi-k3", "k3")],
                ),
            ],
        )
        self.plugin = make_plugin()
        self.plugin._fetch_status = AsyncMock(return_value=self.data)

    def test_overview_lists_models_with_per_cohort_lines(self):
        text = run(command(self.plugin))
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("Mirasim 模型状态 · 可用 "))
        # claude-opus-5-5 is down in 2 pools and ok in 1; the others are ok.
        self.assertIn("/9", lines[0])
        self.assertIn("7/9", lines[0])
        self.assertIn("· claude-opus-5-5 — Claude", text)
        self.assertIn("❌ 体验 · 0% · p50 —", text)
        self.assertIn("❌ 订阅 · 0% · p50 —", text)
        self.assertIn("✅ 云端 · 100% · p50 5.0s", text)
        self.assertIn("· kimi-k3 — Kimi", text)

    def test_problems_sort_first(self):
        text = run(command(self.plugin))
        # claude-opus-5-5 has the worst (down) status and leads the list.
        first_model_idx = text.index("· claude-opus-5-5")
        self.assertLess(first_model_idx, text.index("· kimi-k3"))
        self.assertLess(first_model_idx, text.index("· claude-fable-5-1"))

    def test_configured_models_narrow_the_overview(self):
        self.plugin.config["display_models"] = ["claude-opus-5-5", "gpt-9"]
        text = run(command(self.plugin))
        self.assertIn("claude-opus-5-5", text)
        self.assertNotIn("kimi-k3", text)
        self.assertIn("未找到：gpt-9", text)
        self.assertIn("/mirasim all", text)

    def test_configured_model_with_cohort_suffix_shows_only_that_cohort(self):
        self.plugin.config["display_models"] = ["claude-opus-5-5@paid", "kimi-k3"]
        text = run(command(self.plugin))
        lines = text.splitlines()
        # claude-opus-5-5 narrowed to paid only.
        opus_idx = next(
            i for i, line in enumerate(lines) if line.startswith("· claude-opus-5-5")
        )
        kimi_idx = next(
            i for i, line in enumerate(lines) if line.startswith("· kimi-k3")
        )
        opus_section = lines[opus_idx + 1 : kimi_idx]
        opus_cohort_lines = [
            line
            for line in opus_section
            if line.startswith(("  ✅", "  ⚠️", "  ❌", "  ❔"))
        ]
        self.assertEqual(len(opus_cohort_lines), 1)
        self.assertIn("❌ 订阅 · 0% · p50 —", opus_cohort_lines[0])
        # kimi-k3 keeps all three cohorts; the section ends at the trailing footer.
        kimi_section = "\n".join(lines[kimi_idx:])
        # Header lines should appear 1×; per-cohort rows 3× (free, paid, cloud).
        for label in ("体验", "订阅", "云端"):
            cohort_row = next(
                line for line in lines[kimi_idx:] if f"  ✅ {label} ·" in line
            )
            self.assertIn("100%", cohort_row)
        self.assertEqual(kimi_section.count("  ✅"), 3)

    def test_configured_unknown_cohort_goes_to_missing(self):
        self.plugin.config["display_models"] = ["kimi-k3@bogus", "kimi-k3@paid"]
        text = run(command(self.plugin))
        self.assertIn("未找到：kimi-k3@bogus", text)
        # The valid entry narrows kimi-k3 to paid only.
        self.assertIn("  ✅ 订阅 ·", text)
        self.assertNotIn("  ✅ 体验 ·", text)
        self.assertNotIn("  ✅ 云端 ·", text)

    def test_configured_bare_model_broadens_after_cohort_suffix(self):
        # Listing both forms for the same model: bare wins (shows all cohorts).
        self.plugin.config["display_models"] = ["kimi-k3@paid", "kimi-k3"]
        text = run(command(self.plugin))
        self.assertIn("  ✅ 体验 ·", text)
        self.assertIn("  ✅ 订阅 ·", text)
        self.assertIn("  ✅ 云端 ·", text)

    def test_configured_multiple_cohorts_stack(self):
        self.plugin.config["display_models"] = ["kimi-k3@paid", "kimi-k3@free"]
        text = run(command(self.plugin))
        self.assertIn("  ✅ 体验 ·", text)
        self.assertIn("  ✅ 订阅 ·", text)
        self.assertNotIn("  ✅ 云端 ·", text)

    def test_display_models_order_is_respected(self):
        # Even with kimi-k3 ok and claude-opus-5-5 down, kimi-k3 stays first
        # because that's the order in display_models.
        self.plugin.config["display_models"] = [
            "kimi-k3",
            "claude-opus-5-5",
            "claude-fable-5-1",
        ]
        text = run(command(self.plugin))
        self.assertLess(text.index("· kimi-k3"), text.index("· claude-opus-5-5"))
        self.assertLess(text.index("· claude-opus-5-5"), text.index("· claude-fable-5-1"))

    def test_no_display_models_falls_back_to_status_first(self):
        # display_models=[] (falsy) means "no filter": same default ordering.
        self.plugin.config["display_models"] = []
        text = run(command(self.plugin))
        first_model_idx = text.index("· claude-opus-5-5")
        self.assertLess(first_model_idx, text.index("· kimi-k3"))

    def test_detail_view_default_shows_all_cohorts(self):
        text = run(command(self.plugin, "claude-opus-5-5"))
        self.assertTrue(text.startswith("· claude-opus-5-5"))
        self.assertIn("❌ 体验池 异常（可用率 0%）", text)
        self.assertIn("❌ 订阅池 异常（可用率 0%）", text)
        self.assertIn("✅ 云端池 正常", text)

    def test_detail_view_with_cohort_filter(self):
        text = run(command(self.plugin, "claude-opus-5-5@paid"))
        self.assertTrue(text.startswith("· claude-opus-5-5@paid"))
        self.assertIn("❌ 订阅池 异常", text)
        self.assertNotIn("体验池", text)
        self.assertNotIn("云端池", text)

    def test_detail_view_unknown_cohort(self):
        text = run(command(self.plugin, "claude-opus-5-5@bogus"))
        self.assertIn("未知池「bogus」", text)

    def test_detail_view_shows_failure_reason(self):
        text = run(command(self.plugin, "claude-opus-5-5@free"))
        self.assertIn("限流", text)
        self.assertIn("故障", text)

    def test_fetch_failure_is_reported(self):
        self.plugin._fetch_status = AsyncMock(side_effect=TimeoutError())
        self.assertEqual(
            run(command(self.plugin)), "获取 Mirasim 状态失败：TimeoutError"
        )

    def test_subscription_lifecycle(self):
        plugin = self.plugin
        reply = run(command(plugin, "sub", "opus"))
        self.assertIn("已订阅 claude-opus-5-5 的全部 3 个池", reply)
        self.assertEqual(plugin._subs, {"qq:GroupMessage:1": ["claude-opus-5-5"]})
        plugin.put_kv_data.assert_awaited_with("subscriptions", plugin._subs)

        reply = run(command(plugin, "sub", "claude-fable-5-1@paid"))
        self.assertIn("claude-fable-5-1@paid", reply)
        self.assertIn("订阅池", reply)

        listing = run(command(plugin, "list"))
        self.assertIn("订阅了 2 项", listing)
        self.assertIn("· claude-fable-5-1@paid", listing)

        self.assertIn("还没有订阅", run(command(plugin, "list", umo="qq:Friend:9")))

        reply = run(command(plugin, "unsub", "claude-opus-5-5"))
        self.assertEqual(reply, "已取消订阅 claude-opus-5-5。")
        reply = run(command(plugin, "unsub", "claude-fable-5-1@paid"))
        self.assertEqual(reply, "已取消订阅 claude-fable-5-1@paid。")
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


class ImageTest(unittest.TestCase):
    def setUp(self):
        self.data = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [make_model("<b>odd</b>", "odd")],
                ),
                make_agent(
                    "kimi-code",
                    "Kimi",
                    [make_model("kimi-k3", "k3", status="down", availability=0.0, p50=None)],
                    reasons=[{"class": "outage", "share": 100.0}],
                ),
            ],
            agents_paid=[],
            agents_cloud=[],
        )
        self.plugin = make_plugin(render_image=True)
        self.plugin._fetch_status = AsyncMock(return_value=self.data)

    def fake_image(self, content: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as file:
            file.write(content)
        self.addCleanup(Path(file.name).unlink)
        return file.name

    def rendered_html(self, autoescape: bool) -> str:
        """Render what the plugin sent to T2I with the same Jinja2 engine.

        StrictUndefined turns a misspelt variable into an error here instead
        of a blank spot, or a 500 from T2I and a silent text fallback, later.
        """
        tmpl, tmpl_data = self.plugin.html_render.await_args.args[:2]
        env = jinja2.Environment(
            undefined=jinja2.StrictUndefined, autoescape=autoescape
        )
        return env.from_string(tmpl).render(**tmpl_data)

    def test_overview_is_sent_as_an_image(self):
        path = self.fake_image(b"\x89PNG\r\n\x1a\n...")
        self.plugin.html_render = AsyncMock(return_value=path)
        self.assertEqual(run(command(self.plugin)), ("image", path))
        options = self.plugin.html_render.await_args.kwargs
        self.assertFalse(options["return_url"])
        self.assertEqual(options["options"]["device_scale_factor_level"], "ultra")

    def test_template_renders_the_cards(self):
        self.plugin.html_render = AsyncMock(return_value=self.fake_image(b"\xff\xd8"))
        run(command(self.plugin))

        for autoescape in (False, True):
            html = self.rendered_html(autoescape)
            # The down model comes before the ok one.
            self.assertLess(html.index("kimi-k3"), html.index("&lt;b&gt;odd&lt;/b&gt;"))
            # One row per cohort of the model.
            self.assertIn("体验", html)
            # Strip fills all 48 slots; my fixtures have all-1000 cells.
            self.assertIn('class="ok" style="grid-column: 48"', html)

    def test_text_from_the_api_is_escaped_whatever_t2i_does(self):
        self.plugin.html_render = AsyncMock(return_value=self.fake_image(b"\xff\xd8"))
        run(command(self.plugin, "all"))
        for autoescape in (False, True):
            html = self.rendered_html(autoescape)
            self.assertIn("&lt;b&gt;odd&lt;/b&gt;", html)
            self.assertNotIn("<b>odd</b>", html)

    def test_error_page_from_t2i_falls_back_to_text(self):
        page = self.fake_image(b"<html><title>502 Bad Gateway</title></html>")
        self.plugin.html_render = AsyncMock(return_value=page)
        reply = run(command(self.plugin))
        self.assertTrue(reply.startswith("Mirasim 模型状态 · 可用"))

    def test_template_skips_the_shiki_injection(self):
        # AstrBot otherwise appends a 2.4 MB highlighter to every request.
        template = main_mod.TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertEqual(inject_shiki_runtime(template), template)


if __name__ == "__main__":
    unittest.main()
