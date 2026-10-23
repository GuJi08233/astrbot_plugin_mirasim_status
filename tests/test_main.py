"""Tests for change detection, pushes and commands of the Mirasim plugin.

Run from the AstrBot root so the plugin imports by its package path. Point
ASTRBOT_ROOT elsewhere so importing astrbot.core leaves the real data/ alone:
    ASTRBOT_ROOT=/tmp/mirasim_test uv run python -m unittest \
        data.plugins.astrbot_plugin_mirasim_status.tests.test_main

The fixtures mirror the real payload: ``id`` is only unique inside one agent
(``m1``..``m6`` repeat), the display name is the identity, and the thresholds
block ships with the document.
"""

import asyncio
import importlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

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
    name: str,
    *,
    model_id: str = "m1",
    status: str = "ok",
    availability: float | None = 100.0,
    p50: float | None = 5.0,
    h24: float = 99.0,
    d7: float = 99.0,
    cells: list[int] | None = None,
    same_model: int = 100,
    intel: dict | None = None,
    window: str | None = None,
) -> dict:
    """Build one model entry mirasim.ai-style (a value-per-cell array)."""
    if cells is None:
        cells = [1000] * 48
    return {
        "id": model_id,
        "name": name,
        "now": {"status": status, "availability": availability, "window": window},
        "availability": {"h24": h24, "d7": d7},
        "latency": {"p50": p50, "p95": None},
        "sameModel": same_model,
        "intel": intel or {"full": 100.0, "swapped": 0, "mismatched": 0, "cut": 0},
        "cells": cells,
        "merged": [],
    }


def make_agent(
    agent_id: str,
    name: str,
    models: list[dict],
    *,
    reasons: list[dict] | None = None,
    state: str = "ok",
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
        "state": state,
    }


def make_payload(
    *,
    agents_free: list[dict] | None = None,
    agents_paid: list[dict] | None = None,
    agents_cloud: list[dict] | None = None,
    generated_h: float = 24.0,
    thresholds: dict | None = None,
    notes: list | None = None,
) -> dict:
    """Build the full status payload from per-cohort agent lists."""
    return {
        "schema": 2,
        "generatedAt": iso(generated_h),
        "dataThrough": iso(generated_h),
        "cellSeconds": 1800,
        "cellsStart": iso(0),
        "thresholds": thresholds or {"good": 99, "warn": 95, "minTurns": 20},
        "notes": notes or [],
        "cohorts": [
            {"id": "free", "state": "ok", "agents": agents_free or []},
            {"id": "paid", "state": "ok", "agents": agents_paid or []},
            {"id": "cloud", "state": "ok", "agents": agents_cloud or []},
        ],
    }


class _Context:
    """Minimal stand-in for AstrBot's Context; only send_message is used."""

    def __init__(self) -> None:
        self.send_message = AsyncMock(return_value=True)


def make_plugin(**config) -> MirasimStatus:
    plugin = MirasimStatus(
        _Context(), {"confirm_samples": 2, "render_image": False, **config}
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
    event = AsyncMock()
    event.unified_msg_origin = umo
    event.plain_result = lambda text: text
    event.image_result = lambda path: ("image", path)
    replies = [reply async for reply in plugin.cmd_mirasim(event, *args)]
    return replies[0]


class SlugTest(unittest.TestCase):
    """Model names carry dots and spaces; the key is a slug of the name."""

    def test_names_slug_to_stable_keys(self):
        self.assertEqual(main_mod._slug("opus 5.5"), "opus-5-5")
        self.assertEqual(main_mod._slug("gpt-6.1-sol"), "gpt-6-1-sol")
        self.assertEqual(main_mod._slug("Gemini 4 Argon"), "gemini-4-argon")
        self.assertEqual(main_mod._slug("GLM-5.3-Flash"), "glm-5-3-flash")

    def test_empty_and_non_string_input(self):
        self.assertEqual(main_mod._slug(None), "")
        self.assertEqual(main_mod._slug("  "), "")
        self.assertEqual(main_mod._slug("!!!"), "")


class CollectModelsTest(unittest.TestCase):
    """The payload's ``id`` repeats per agent, so grouping must use the name."""

    def payload(self) -> dict:
        # Both agents call their first model "m1"; only the name tells them apart.
        return make_payload(
            agents_free=[
                make_agent("claude-code", "Claude", [make_model("opus 5.5", model_id="m1")]),
                make_agent("codex", "GPT", [make_model("gpt-6-astra", model_id="m1")]),
            ],
            agents_paid=[
                make_agent("claude-code", "Claude", [make_model("opus 5.5", model_id="m1")]),
                make_agent("codex", "GPT", [make_model("gpt-6-astra", model_id="m1")]),
            ],
            agents_cloud=[],
        )

    def test_colliding_ids_stay_separate_models(self):
        models = main_mod._collect_models(self.payload())
        self.assertEqual(sorted(models), ["gpt-6-astra", "opus-5-5"])
        self.assertEqual(len(models["opus-5-5"]["entries"]), 2)
        self.assertEqual(models["opus-5-5"]["agent"], "Claude")
        self.assertEqual(models["gpt-6-astra"]["agent"], "GPT")

    def test_entries_are_ordered_by_cohort(self):
        models = main_mod._collect_models(self.payload())
        cohorts = [c for c, _m, _a in models["opus-5-5"]["entries"]]
        self.assertEqual(cohorts, ["free", "paid"])

    def test_missing_name_falls_back_to_id(self):
        data = make_payload(
            agents_free=[make_agent("a", "A", [make_model("", model_id="only-id")])]
        )
        models = main_mod._collect_models(data)
        self.assertIn("only-id", models)


class StatusReadingTest(unittest.TestCase):
    """``now.status`` is authoritative; availability only fills gaps."""

    def test_known_statuses_are_taken_verbatim(self):
        for raw in ("ok", "degraded", "down"):
            model = make_model("m", status=raw, availability=50.0)
            self.assertEqual(main_mod._model_status(model)[0], raw)

    def test_status_vocabulary_aliases(self):
        for raw, expected in (
            ("good", "ok"),
            ("warn", "degraded"),
            ("warning", "degraded"),
            ("outage", "down"),
            ("no_data", "nodata"),
        ):
            model = make_model("m", status=raw, availability=90.0)
            self.assertEqual(main_mod._model_status(model)[0], expected)

    def test_unknown_status_falls_back_to_payload_thresholds(self):
        data = make_payload(thresholds={"good": 99, "warn": 95})
        cases = {
            100.0: "ok",
            99.0: "ok",
            98.9: "degraded",
            95.0: "degraded",
            94.9: "down",
            60.0: "down",
        }
        for availability, expected in cases.items():
            model = make_model("m", status="weird", availability=availability)
            self.assertEqual(
                main_mod._model_status(model, data)[0], expected, availability
            )

    def test_missing_availability_is_nodata(self):
        model = make_model("m", status="weird", availability=None)
        self.assertEqual(main_mod._model_status(model)[0], "nodata")

    def test_thresholds_block_is_honoured(self):
        data = make_payload(thresholds={"good": 80, "warn": 60})
        model = make_model("m", status="weird", availability=70.0)
        self.assertEqual(main_mod._model_status(model, data)[0], "degraded")

    def test_missing_thresholds_block_uses_defaults(self):
        good, warn, min_turns = main_mod._thresholds({"cohorts": []})
        self.assertEqual((good, warn), (99.0, 95.0))
        self.assertIsNone(min_turns)

    def test_inverted_thresholds_are_clamped(self):
        good, warn, _ = main_mod._thresholds({"thresholds": {"good": 90, "warn": 99}})
        self.assertEqual((good, warn), (90.0, 90.0))

    def test_agent_status_prefers_its_summary(self):
        data = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [make_model("opus 5.5", status="ok", availability=100.0)],
                    state="degraded",
                )
            ]
        )
        agent = data["cohorts"][0]["agents"][0]
        # summary mirrors the first model, so it also says ok here.
        self.assertEqual(main_mod._agent_status(agent, data)[0], "ok")

    def test_agent_status_falls_back_to_models(self):
        agent = {"id": "a", "name": "A", "summary": {}, "models": [make_model("m")]}
        self.assertEqual(main_mod._agent_status(agent)[0], "ok")


class HistoryTest(unittest.TestCase):
    """Cells hold availability ×10 and are bucketed by the payload thresholds."""

    def test_cell_buckets(self):
        self.assertEqual(main_mod._cell_status(1000, 99, 95), "ok")
        self.assertEqual(main_mod._cell_status(990, 99, 95), "ok")
        self.assertEqual(main_mod._cell_status(989, 99, 95), "degraded")
        self.assertEqual(main_mod._cell_status(950, 99, 95), "degraded")
        self.assertEqual(main_mod._cell_status(949, 99, 95), "down")
        self.assertEqual(main_mod._cell_status(-1, 99, 95), "unknown")
        self.assertEqual(main_mod._cell_status(None, 99, 95), "unknown")

    def test_history_bars_use_payload_thresholds(self):
        cells = [700] * 48  # 70% everywhere: under the site's own 99/95.
        data = make_payload(thresholds={"good": 99, "warn": 95})
        self.assertEqual(main_mod._history_bars(cells, data), ["down"] * 48)

    def test_worst_status_ordering(self):
        self.assertEqual(main_mod._worst_status(["ok", "down"]), "down")
        self.assertEqual(main_mod._worst_status(["ok", "degraded"]), "degraded")
        self.assertEqual(main_mod._worst_status([]), "unknown")

    def test_last_streak_locates_the_trailing_run(self):
        bars = ["ok"] * 10 + ["down"] * 3 + ["ok"] * 2
        self.assertEqual(main_mod._last_streak(bars, ("down",)), (10, 3))
        self.assertEqual(main_mod._last_streak(["ok"] * 4, ("down",)), None)


class FormatTest(unittest.TestCase):
    def test_rate_and_seconds(self):
        self.assertEqual(main_mod._fmt_rate(100.0, short=True), "100%")
        self.assertEqual(main_mod._fmt_rate(97.94), "97.9%")
        self.assertEqual(main_mod._fmt_rate(None), "—")
        self.assertEqual(main_mod._fmt_seconds(4.14), "4.1s")
        self.assertEqual(main_mod._fmt_seconds(None), "—")

    def test_duration_drops_empty_units(self):
        self.assertEqual(main_mod._fmt_duration(3600), "1小时")
        self.assertEqual(main_mod._fmt_duration(5400), "1小时30分钟")
        self.assertEqual(main_mod._fmt_duration(86400), "1天")
        self.assertEqual(main_mod._fmt_duration(90), "1分钟")

    def test_window_units(self):
        self.assertEqual(main_mod._fmt_window("15m"), "15分钟")
        self.assertEqual(main_mod._fmt_window("1h"), "1小时")
        self.assertEqual(main_mod._fmt_window(None), "")
        self.assertEqual(main_mod._fmt_window("weird"), "weird")

    def test_reasons_render_bare_then_bracketed(self):
        reasons = [
            {"class": "capacity", "share": 98},
            {"class": "outage", "share": 2},
            {"class": "throttle", "share": 0},
        ]
        self.assertEqual(main_mod._format_reasons(reasons), "容量不足98%·故障2%")
        self.assertEqual(main_mod._reason_note(reasons), "（容量不足98%·故障2%）")
        self.assertEqual(main_mod._format_reasons(None), "")


class ResolveKeyTest(unittest.TestCase):
    KEYS = ["opus-5", "opus-5-5", "sonnet-5-5", "k3", "gpt-6-astra"]

    def test_exact_match_beats_longer_keys(self):
        self.assertEqual(main_mod._resolve_key(self.KEYS, "opus-5", ""), "opus-5")

    def test_spaces_and_dots_are_forgiven(self):
        self.assertEqual(main_mod._resolve_key(self.KEYS, "opus 5.5", ""), "opus-5-5")
        self.assertEqual(main_mod._resolve_key(self.KEYS, "OPUS 5", ""), "opus-5")

    def test_ambiguous_input_lists_the_candidates(self):
        with self.assertRaises(LookupError) as ctx:
            main_mod._resolve_key(self.KEYS, "opus", "")
        self.assertIn("opus-5、opus-5-5", str(ctx.exception))

    def test_unknown_input_is_reported(self):
        with self.assertRaises(LookupError) as ctx:
            main_mod._resolve_key(self.KEYS, "claude-opus", "看全部。")
        self.assertIn("看全部。", str(ctx.exception))


class LegacyNameTest(unittest.TestCase):
    """The old vendor-prefixed names still resolve to the current keys."""

    def setUp(self):
        self.data = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [make_model("opus 5.5"), make_model("sonnet 5.5")],
                ),
                make_agent("gemini-cli", "Gemini", [make_model("Gemini 4 Argon")]),
            ]
        )
        self.models = main_mod._collect_models(self.data)
        self.aliases = main_mod._model_aliases(self.models)

    def test_aliases_carry_the_vendor_prefix(self):
        self.assertEqual(self.aliases["claude-opus-5-5"], "opus-5-5")
        self.assertEqual(self.aliases["claude-sonnet-5-5"], "sonnet-5-5")
        self.assertEqual(self.aliases["gemini-gemini-4-argon"], "gemini-4-argon")

    def test_alias_resolves_the_old_spelling(self):
        keys = list(self.models)
        self.assertEqual(
            main_mod._resolve_key(keys, "claude-opus-5-5", "", self.aliases),
            "opus-5-5",
        )
        self.assertEqual(
            main_mod._resolve_key(keys, "Claude Opus 5.5", "", self.aliases),
            "opus-5-5",
        )

    def test_current_names_still_win(self):
        keys = list(self.models)
        self.assertEqual(
            main_mod._resolve_key(keys, "opus-5-5", "", self.aliases), "opus-5-5"
        )

    def test_old_subscription_covers_the_new_change_key(self):
        self.assertTrue(
            main_mod._covers("claude-opus-5-5", "opus-5-5@free", self.aliases)
        )
        self.assertTrue(
            main_mod._covers("claude-opus-5-5@paid", "opus-5-5@paid", self.aliases)
        )
        self.assertFalse(
            main_mod._covers("claude-opus-5-5@paid", "opus-5-5@free", self.aliases)
        )
        self.assertFalse(
            main_mod._covers("claude-sonnet-5-5", "opus-5-5@free", self.aliases)
        )

    def test_display_models_accepts_the_old_spelling(self):
        plugin = make_plugin(display_models=["claude-opus-5-5"])
        plugin._fetch_status = AsyncMock(return_value=self.data)
        groups, missing, filtered = plugin._overview_groups(self.data, show_all=False)
        self.assertTrue(filtered)
        self.assertEqual(missing, [])
        self.assertEqual([g["agent"] for g in groups], ["Claude"])
        self.assertEqual([m["key"] for m in groups[0]["models"]], ["opus-5-5"])


class VendorFilterTest(unittest.TestCase):
    """A configured entry may name a vendor instead of every single model."""

    def setUp(self):
        self.data = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [make_model("opus 5.5"), make_model("sonnet 5.5")],
                ),
                make_agent(
                    "codex", "GPT", [make_model("gpt-6-astra"), make_model("gpt-6-luna")]
                ),
                make_agent("dsh", "DeepSeek", [make_model("v4.1-flash")]),
            ],
            # Only one GPT model is served from the paid pool.
            agents_paid=[make_agent("codex", "GPT", [make_model("gpt-6-astra")])],
        )

    def overview(self, display_models):
        plugin = make_plugin(display_models=display_models)
        plugin._fetch_status = AsyncMock(return_value=self.data)
        return plugin._overview_groups(self.data, show_all=False)

    def test_vendor_name_selects_every_model_of_that_vendor(self):
        groups, missing, filtered = self.overview(["gpt"])
        self.assertEqual(missing, [])
        self.assertEqual([g["agent"] for g in groups], ["GPT"])
        self.assertEqual(
            sorted(m["key"] for m in groups[0]["models"]),
            ["gpt-6-astra", "gpt-6-luna"],
        )

    def test_vendor_id_also_works(self):
        groups, missing, _ = self.overview(["dsh"])
        self.assertEqual(missing, [])
        self.assertEqual([m["key"] for m in groups[0]["models"]], ["v4-1-flash"])

    def test_two_vendors_can_be_mixed_with_a_model(self):
        groups, missing, _ = self.overview(["deepseek", "opus-5-5"])
        self.assertEqual(missing, [])
        self.assertEqual(sorted(g["agent"] for g in groups), ["Claude", "DeepSeek"])

    def test_vendor_with_a_cohort_limits_the_pools(self):
        groups, missing, _ = self.overview(["gpt@paid"])
        self.assertEqual(missing, [])
        self.assertEqual([g["agent"] for g in groups], ["GPT"])
        # Only the model that serves the paid pool is kept.
        self.assertEqual([m["key"] for m in groups[0]["models"]], ["gpt-6-astra"])
        self.assertEqual(groups[0]["cohorts"], ["paid"])

    def test_vendor_with_a_cohort_nobody_serves_is_reported(self):
        groups, missing, _ = self.overview(["deepseek@paid"])
        self.assertEqual(groups, [])
        self.assertEqual(missing, ["deepseek@paid"])

    def test_ambiguous_partial_name_is_reported(self):
        groups, missing, _ = self.overview(["gpt-6"])
        self.assertEqual(groups, [])
        self.assertEqual(missing, ["gpt-6"])

    def test_unknown_entry_is_reported(self):
        groups, missing, _ = self.overview(["no-such-model"])
        self.assertEqual(groups, [])
        self.assertEqual(missing, ["no-such-model"])


class TargetSplitTest(unittest.TestCase):
    def test_plain_and_suffixed_forms(self):
        self.assertEqual(main_mod._split_target("opus 5.5"), ("opus 5.5", None))
        self.assertEqual(main_mod._split_target("opus 5.5@paid"), ("opus 5.5", "paid"))

    def test_cohort_aliases_are_normalised(self):
        self.assertEqual(main_mod._split_target("k3@订阅")[1], "paid")
        self.assertEqual(main_mod._split_target("k3@云端")[1], "cloud")
        self.assertEqual(main_mod._split_target("k3@免费")[1], "free")
        self.assertEqual(main_mod._split_target("k3@体验")[1], "free")

    def test_trailing_at_is_ignored(self):
        self.assertEqual(main_mod._split_target("k3@"), ("k3", None))

    def test_covers_matches_model_and_cohort(self):
        self.assertTrue(main_mod._covers("*", "opus-5-5@free"))
        self.assertTrue(main_mod._covers("opus-5-5", "opus-5-5@free"))
        self.assertTrue(main_mod._covers("opus-5-5@paid", "opus-5-5@paid"))
        self.assertFalse(main_mod._covers("opus-5-5@paid", "opus-5-5@free"))
        self.assertFalse(main_mod._covers("k3", "opus-5-5@free"))


class ChangeDetectionTest(unittest.TestCase):
    """The plugin watches status across two consecutive polls."""

    def _payload_with(self, states: dict[str, str]) -> dict:
        """One Claude model across all three pools in the given states."""
        availability = {
            "ok": 100.0,
            "degraded": 97.5,
            "down": 40.0,
        }
        agents = {}
        for cohort in ("free", "paid", "cloud"):
            agents[cohort] = [
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "opus 5.5",
                            status=states[cohort],
                            availability=availability[states[cohort]],
                        )
                    ],
                )
            ]
        return make_payload(
            agents_free=agents["free"],
            agents_paid=agents["paid"],
            agents_cloud=agents["cloud"],
        )

    def test_first_poll_records_a_silent_baseline(self):
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        data = self._payload_with({"free": "down", "paid": "degraded", "cloud": "ok"})
        run(plugin._check_changes(data))
        self.assertEqual(
            plugin._states,
            {
                "opus-5-5@free": "down",
                "opus-5-5@paid": "degraded",
                "opus-5-5@cloud": "ok",
            },
        )
        plugin.put_kv_data.assert_awaited_once_with("states", plugin._states)
        plugin.context.send_message.assert_not_awaited()

    def test_single_poll_flip_does_not_push_with_confirm_two(self):
        """A change lasting one poll is just a blip."""
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {
            "opus-5-5@free": "ok",
            "opus-5-5@paid": "ok",
            "opus-5-5@cloud": "ok",
        }
        prev = self._payload_with({"free": "ok", "paid": "ok", "cloud": "ok"})
        cur = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        run(plugin._check_changes(cur, prev))
        plugin.context.send_message.assert_not_awaited()

    def test_confirmed_failure_pushes_per_cohort(self):
        """A failure seen on two consecutive polls fires."""
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {
            "opus-5-5@free": "ok",
            "opus-5-5@paid": "ok",
            "opus-5-5@cloud": "ok",
        }
        prev = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        cur = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        run(plugin._check_changes(cur, prev))
        sent = sent_messages(plugin)
        self.assertIn("🔴 opus 5.5（免费）异常", sent["s"])
        self.assertNotIn("（订阅）异常", sent["s"])
        self.assertEqual(plugin._states["opus-5-5@free"], "down")

    def test_degraded_is_its_own_push_wording(self):
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {
            "opus-5-5@free": "ok",
            "opus-5-5@paid": "ok",
            "opus-5-5@cloud": "ok",
        }
        prev = self._payload_with({"free": "degraded", "paid": "ok", "cloud": "ok"})
        cur = self._payload_with({"free": "degraded", "paid": "ok", "cloud": "ok"})
        run(plugin._check_changes(cur, prev))
        sent = sent_messages(plugin)
        self.assertIn("🟡 opus 5.5（免费）降级（可用率 97.5%）", sent["s"])

    def test_only_subscribed_cohort_gets_the_push(self):
        plugin = make_plugin()
        plugin._subs = {
            "s-paid": ["opus-5-5@paid"],
            "s-free": ["opus-5-5@free"],
            "s-all-cohorts": ["opus-5-5"],
        }
        plugin._states = {
            "opus-5-5@free": "ok",
            "opus-5-5@paid": "ok",
            "opus-5-5@cloud": "ok",
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
            "opus-5-5@free": "ok",
            "opus-5-5@paid": "ok",
            "opus-5-5@cloud": "ok",
        }
        cur = self._payload_with({"free": "down", "paid": "ok", "cloud": "ok"})
        run(plugin._check_changes(cur, None))
        self.assertIn("🔴 opus 5.5（免费）异常", sent_messages(plugin)["s"])

    def test_recovery_lists_outage_duration(self):
        """The ok push reports how long the failure run lasted."""
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {"opus-5-5@free": "down"}
        # 41 ok cells, then 6 down cells (3 hours), then 1 ok cell.
        cells = [1000] * 41 + [400] * 6 + [1000]
        prev = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "opus 5.5",
                            status="ok",
                            availability=100.0,
                            p50=4.2,
                            cells=cells,
                        )
                    ],
                )
            ]
        )
        cur = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model(
                            "opus 5.5",
                            status="ok",
                            availability=100.0,
                            p50=4.2,
                            cells=cells,
                        )
                    ],
                )
            ]
        )
        run(plugin._check_changes(cur, prev))
        sent = sent_messages(plugin)["s"]
        self.assertIn("🟢 opus 5.5（免费）已恢复", sent)
        self.assertIn("p50 4.2s", sent)
        self.assertIn("异常持续约 3小时", sent)

    def test_restart_restores_baseline_without_pushing(self):
        """States loaded from KV act as the baseline, so no stale flood."""
        plugin = make_plugin()
        plugin._subs = {"s": ["*"]}
        plugin._states = {
            "opus-5-5@free": "down",
            "opus-5-5@paid": "down",
            "opus-5-5@cloud": "down",
        }
        data = self._payload_with({"free": "down", "paid": "down", "cloud": "down"})
        run(plugin._check_changes(data, data))
        plugin.context.send_message.assert_not_awaited()


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.data = make_payload(
            agents_free=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model("opus 5.5", status="down", availability=40.0, p50=None),
                        make_model("fable 5.1", model_id="m2"),
                    ],
                    reasons=[
                        {"class": "throttle", "share": 78.5},
                        {"class": "outage", "share": 21.4},
                    ],
                ),
                make_agent(
                    "codex",
                    "GPT",
                    [make_model("gpt-6-astra", status="degraded", availability=98.4)],
                ),
                make_agent("kimi-code", "Kimi", [make_model("k3", model_id="m1")]),
            ],
            agents_paid=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model("opus 5.5", status="down", availability=40.0, p50=None),
                        make_model("fable 5.1", model_id="m2"),
                    ],
                ),
                make_agent("kimi-code", "Kimi", [make_model("k3", model_id="m1")]),
            ],
            agents_cloud=[
                make_agent(
                    "claude-code",
                    "Claude",
                    [
                        make_model("opus 5.5"),
                        make_model("fable 5.1", model_id="m2"),
                    ],
                ),
                make_agent("kimi-code", "Kimi", [make_model("k3", model_id="m1")]),
            ],
        )
        self.plugin = make_plugin()
        self.plugin._fetch_status = AsyncMock(return_value=self.data)

    def test_overview_groups_by_vendor_with_pool_segments(self):
        text = run(command(self.plugin))
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("Mirasim 模型状态 · 正常 "))
        # 10 rows: opus/fable/k3 in 3 pools each, gpt in free only; 7 are ok
        # (opus is down in free+paid and ok in cloud, gpt is degraded).
        self.assertIn("正常 7/10", lines[0])
        self.assertIn("Claude · 4/6 正常", text)
        self.assertIn("· opus 5.5 — ❌免费 40% · ❌订阅 40% · ✅云端 100%", text)
        self.assertIn("· gpt-6-astra — ⚠️免费 98.4%", text)

    def test_problems_sort_vendors_first(self):
        text = run(command(self.plugin))
        # Claude has a down row and leads; Kimi is entirely ok and comes last.
        self.assertLess(text.index("Claude ·"), text.index("Kimi ·"))

    def test_configured_models_narrow_the_overview(self):
        self.plugin.config["display_models"] = ["opus 5.5", "gpt-9"]
        text = run(command(self.plugin))
        self.assertIn("opus 5.5", text)
        self.assertNotIn("k3", text)
        self.assertIn("未找到：gpt-9", text)
        self.assertIn("/mirasim all", text)

    def test_configured_model_with_cohort_suffix_shows_only_that_cohort(self):
        self.plugin.config["display_models"] = ["opus 5.5@paid", "k3"]
        text = run(command(self.plugin))
        opus_line = next(
            line for line in text.splitlines() if line.startswith("· opus 5.5")
        )
        # One pool shown, so the line holds a single segment separator.
        self.assertEqual(opus_line.count("·"), 1)
        self.assertIn("❌订阅 40%", opus_line)
        self.assertNotIn("免费", opus_line)
        self.assertNotIn("云端", opus_line)
        kimi_line = next(line for line in text.splitlines() if line.startswith("· k3"))
        self.assertEqual(kimi_line.count("·"), 3)

    def test_configured_unknown_cohort_goes_to_missing(self):
        self.plugin.config["display_models"] = ["k3@bogus", "k3@paid"]
        text = run(command(self.plugin))
        self.assertIn("未找到：k3@bogus", text)
        self.assertIn("· k3 — ✅订阅 100%", text)

    def test_configured_bare_model_broadens_after_cohort_suffix(self):
        self.plugin.config["display_models"] = ["k3@paid", "k3"]
        text = run(command(self.plugin))
        self.assertIn("· k3 — ✅免费 100% · ✅订阅 100% · ✅云端 100%", text)

    def test_configured_multiple_cohorts_stack(self):
        self.plugin.config["display_models"] = ["k3@paid", "k3@free"]
        text = run(command(self.plugin))
        self.assertIn("· k3 — ✅免费 100% · ✅订阅 100%", text)
        self.assertNotIn("云端", text)

    def test_display_models_order_is_respected(self):
        self.plugin.config["display_models"] = ["k3", "opus 5.5", "fable 5.1"]
        text = run(command(self.plugin))
        self.assertLess(text.index("· k3"), text.index("· opus 5.5"))
        self.assertLess(text.index("· opus 5.5"), text.index("· fable 5.1"))

    def test_detail_view_default_shows_all_cohorts(self):
        text = run(command(self.plugin, "opus 5.5"))
        self.assertTrue(text.startswith("· opus 5.5（Claude）"))
        self.assertIn("❌ 免费池 异常", text)
        self.assertIn("❌ 订阅池 异常", text)
        self.assertIn("✅ 云端池 正常", text)

    def test_detail_view_with_cohort_filter(self):
        text = run(command(self.plugin, "opus 5.5@paid"))
        self.assertTrue(text.startswith("· opus 5.5@paid"))
        self.assertIn("❌ 订阅池 异常", text)
        self.assertNotIn("免费池", text)
        self.assertNotIn("云端池", text)

    def test_detail_view_accepts_chinese_cohort(self):
        text = run(command(self.plugin, "opus 5.5@订阅"))
        self.assertTrue(text.startswith("· opus 5.5@paid"))
        self.assertIn("❌ 订阅池 异常", text)

    def test_detail_view_unknown_cohort(self):
        text = run(command(self.plugin, "opus 5.5@bogus"))
        self.assertIn("未知池「bogus」", text)

    def test_detail_view_shows_failure_reason(self):
        text = run(command(self.plugin, "opus 5.5@free"))
        self.assertIn("限流79%·故障21%", text)

    def test_model_name_with_spaces_works(self):
        """GreedyStr keeps "opus 5.5" together instead of splitting on space."""
        text = run(command(self.plugin, "opus", "5.5"))
        self.assertTrue(text.startswith("· opus 5.5"))

    def test_agents_view_reads_summaries(self):
        text = run(command(self.plugin, "agents"))
        self.assertIn("Mirasim 厂商状态", text)
        self.assertIn("· Claude", text)
        self.assertIn("❌ 免费", text)
        self.assertIn("· Kimi", text)

    def test_help_lists_commands(self):
        text = run(command(self.plugin, "help"))
        self.assertIn("/mirasim all", text)
        self.assertIn("/mirasim agents", text)

    def test_fetch_failure_is_reported(self):
        self.plugin._fetch_status = AsyncMock(side_effect=TimeoutError())
        self.assertEqual(
            run(command(self.plugin)), "获取 Mirasim 状态失败：TimeoutError"
        )

    def test_subscription_lifecycle(self):
        plugin = self.plugin
        reply = run(command(plugin, "sub", "opus 5.5"))
        self.assertIn("已订阅 opus 5.5 的全部 3 个池", reply)
        self.assertEqual(plugin._subs, {"qq:GroupMessage:1": ["opus-5-5"]})
        plugin.put_kv_data.assert_awaited_with("subscriptions", plugin._subs)

        reply = run(command(plugin, "sub", "fable 5.1@paid"))
        self.assertIn("fable 5.1@paid", reply)
        self.assertIn("订阅池", reply)

        listing = run(command(plugin, "list"))
        self.assertIn("订阅了 2 项", listing)
        self.assertIn("· fable-5-1@paid", listing)

        self.assertIn("还没有订阅", run(command(plugin, "list", umo="qq:Friend:9")))

        self.assertEqual(
            run(command(plugin, "unsub", "opus 5.5")), "已取消订阅 opus-5-5。"
        )
        self.assertEqual(
            run(command(plugin, "unsub", "fable 5.1@paid")),
            "已取消订阅 fable-5-1@paid。",
        )
        self.assertEqual(plugin._subs, {})

    def test_duplicate_subscription_is_reported(self):
        plugin = self.plugin
        run(command(plugin, "sub", "opus 5.5"))
        self.assertIn("已订阅", run(command(plugin, "sub", "opus 5.5")))

    def test_broad_subscription_covers_narrow_target(self):
        plugin = self.plugin
        run(command(plugin, "sub", "opus 5.5"))
        self.assertIn("已覆盖", run(command(plugin, "sub", "opus 5.5@paid")))

    def test_subscribing_to_everything(self):
        plugin = self.plugin
        run(command(plugin, "sub", "opus 5.5"))
        self.assertIn("已订阅全部模型", run(command(plugin, "sub", "all")))
        self.assertEqual(plugin._subs, {"qq:GroupMessage:1": ["*"]})
        self.assertIn("无需单独订阅", run(command(plugin, "sub", "k3")))
        self.assertIn("unsub all", run(command(plugin, "unsub", "k3")))
        self.assertEqual(
            run(command(plugin, "unsub", "all")), "已取消本会话的全部订阅。"
        )
        self.assertEqual(plugin._subs, {})


class ImageTest(unittest.TestCase):
    def setUp(self):
        self.data = make_payload(
            agents_free=[
                make_agent("claude-code", "Claude", [make_model("<b>odd</b>")]),
                make_agent(
                    "kimi-code",
                    "Kimi",
                    [make_model("k3", status="down", availability=40.0, p50=None)],
                    reasons=[{"class": "outage", "share": 100.0}],
                ),
            ]
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

    def test_template_renders_groups_and_pools(self):
        self.plugin.html_render = AsyncMock(return_value=self.fake_image(b"\xff\xd8"))
        run(command(self.plugin))
        for autoescape in (False, True):
            html = self.rendered_html(autoescape)
            # The down vendor comes first, and both vendors are labelled.
            self.assertLess(html.index("Kimi"), html.index("&lt;b&gt;odd&lt;/b&gt;"))
            self.assertIn("Claude", html)
            # One pool row per cohort.
            self.assertIn("免费", html)
            # Strip fills all 48 slots; the fixtures use all-1000 cells.
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
        self.assertTrue(reply.startswith("Mirasim 模型状态 · 正常"))

    def test_template_skips_the_shiki_injection(self):
        # AstrBot otherwise appends a 2.4 MB highlighter to every request.
        template = main_mod.TEMPLATE_PATH.read_text(encoding="utf-8")
        self.assertEqual(inject_shiki_runtime(template), template)


if __name__ == "__main__":
    unittest.main()
