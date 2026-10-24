"""Tests for the price list read off the Mirasim price page.

Run from the AstrBot root so the plugin imports by its package path. Point
ASTRBOT_ROOT elsewhere so importing astrbot.core leaves the real data/ alone:
    ASTRBOT_ROOT=/tmp/mirasim_test uv run python -m unittest \
        data.plugins.astrbot_plugin_mirasim_status.tests.test_pricing
"""

import unittest

from data.plugins.astrbot_plugin_mirasim_status import main as main_mod
from data.plugins.astrbot_plugin_mirasim_status import pricing

PAGE = """
<html><body>
<p>你买的是额度，各模型按刊例价扣；打折是限时优惠。</p>
<article class="mp-card" aria-labelledby="mp-v-claude">
  <header class="mp-card-head">
    <h3 class="mp-vendor" id="mp-v-claude"><span class="mp-mark"><svg><title>Claude</title></svg></span>Claude</h3>
  </header>
  <table class="mp-table">
    <thead><tr><th scope="col">模型</th><th scope="col">输入</th><th scope="col">输出</th><th scope="col">缓存读取</th></tr></thead>
    <tbody>
      <tr><th scope="row">Claude Opus 5.5</th><td>4</td><td>20</td><td>0.2</td></tr>
    </tbody>
  </table>
</article>
<article class="mp-card" aria-labelledby="mp-v-openai">
  <header class="mp-card-head">
    <h3 class="mp-vendor" id="mp-v-openai"><span class="mp-mark"><svg><title>OpenAI</title></svg></span>OpenAI</h3>
    <span class="mp-badge">限时 1 折</span>
  </header>
  <table class="mp-table">
    <tbody>
      <tr><th scope="row">GPT-6 Astra</th><td><s class="mp-was" aria-label="刊例价 10">10</s>1</td><td><s class="mp-was" aria-label="刊例价 50">50</s>5</td><td><s class="mp-was" aria-label="刊例价 1">1</s>0.1</td></tr>
      <tr><th scope="row">GPT-6 Luna</th><td><s class="mp-was" aria-label="刊例价 0.1">0.1</s>0.01</td><td><s class="mp-was" aria-label="刊例价 0.5">0.5</s>0.05</td><td><s class="mp-was" aria-label="刊例价 0.01">0.01</s>0.001</td></tr>
    </tbody>
  </table>
</article>
<article class="mp-card" aria-labelledby="mp-v-kimi">
  <header class="mp-card-head">
    <h3 class="mp-vendor" id="mp-v-kimi">Kimi</h3>
  </header>
  <table class="mp-table">
    <tbody>
      <tr><th scope="row">Kimi K3<span class="mp-tag">限时 1 折</span></th><td><s class="mp-was" aria-label="刊例价 3">3</s>0.3</td><td>1.5</td><td>—</td></tr>
    </tbody>
  </table>
</article>
<article class="mp-card" aria-labelledby="mp-v-deepseek">
  <header class="mp-card-head">
    <h3 class="mp-vendor" id="mp-v-deepseek">DeepSeek</h3>
  </header>
  <table class="mp-table">
    <tbody>
      <tr><th scope="row">DeepSeek V4.1 Flash</th><td>0.3</td><td>1.2</td><td>0.006</td></tr>
    </tbody>
  </table>
</article>
<p>缓存写入也按官方 API 价扣；用自己的账号或 API 密钥不扣额度。更新于 2026-10-24</p>
</body></html>
"""


class ParseTest(unittest.TestCase):
    def setUp(self):
        self.prices = pricing.parse_prices(PAGE)

    def test_every_model_is_read(self):
        self.assertEqual(len(self.prices.entries), 5)

    def test_plain_prices(self):
        entry = self.prices.entries["claude-opus-5-5"]
        self.assertEqual(entry.name, "Claude Opus 5.5")
        self.assertEqual(entry.vendor, "Claude")
        self.assertEqual(entry.input_price, (None, 4.0))
        self.assertEqual(entry.output_price, (None, 20.0))
        self.assertEqual(entry.cache_read_price, (None, 0.2))
        self.assertFalse(entry.discounted)

    def test_struck_price_keeps_both_numbers(self):
        entry = self.prices.entries["gpt-6-astra"]
        self.assertEqual(entry.input_price, (10.0, 1.0))
        self.assertEqual(entry.output_price, (50.0, 5.0))
        self.assertEqual(entry.cache_read_price, (1.0, 0.1))
        self.assertTrue(entry.discounted)

    def test_the_vendor_badge_applies_to_its_models(self):
        self.assertEqual(self.prices.entries["gpt-6-luna"].badge, "限时 1 折")

    def test_a_row_tag_does_not_stick_to_the_name(self):
        entry = self.prices.entries["kimi-k3"]
        self.assertEqual(entry.name, "Kimi K3")
        self.assertEqual(entry.tag, "限时 1 折")

    def test_a_missing_price_is_none(self):
        entry = self.prices.entries["kimi-k3"]
        self.assertEqual(entry.cache_read_price, (None, None))

    def test_the_update_date_is_read(self):
        self.assertEqual(self.prices.updated, "2026-10-24")


class FindTest(unittest.TestCase):
    def setUp(self):
        self.prices = pricing.parse_prices(PAGE)

    def test_the_vendor_prefixed_name_is_found(self):
        entry = self.prices.find("Claude", "opus 5.5")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "Claude Opus 5.5")

    def test_the_bare_model_name_is_found(self):
        entry = self.prices.find("GPT", "gpt-6-astra")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "GPT-6 Astra")

    def test_a_dated_snapshot_falls_back_to_its_model(self):
        entry = self.prices.find("DeepSeek", "v4.1-flash")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.name, "DeepSeek V4.1 Flash")

    def test_a_version_number_is_not_mistaken_for_a_snapshot(self):
        # "opus 5.5" must not fall back onto "opus 5".
        self.assertIsNone(self.prices.find("Claude", "opus 5"))
        self.assertIsNotNone(self.prices.find("Claude", "opus 5.5"))

    def test_an_unknown_model_is_not_found(self):
        self.assertIsNone(self.prices.find("Nobody", "nothing"))


class PriceBitsTest(unittest.TestCase):
    def setUp(self):
        self.prices = pricing.parse_prices(PAGE)

    def test_plain_prices_have_no_list_price(self):
        entry = self.prices.entries["claude-opus-5-5"]
        self.assertEqual(
            main_mod._price_bits(entry), "输入 4 · 输出 20 · 缓存读取 0.2"
        )

    def test_a_discounted_price_shows_the_list_price(self):
        entry = self.prices.entries["gpt-6-astra"]
        self.assertEqual(
            main_mod._price_bits(entry),
            "输入 1（刊例 10） · 输出 5（刊例 50） · 缓存读取 0.1（刊例 1）",
        )

    def test_a_missing_price_shows_a_dash(self):
        entry = self.prices.entries["kimi-k3"]
        self.assertEqual(
            main_mod._price_bits(entry),
            "输入 0.3（刊例 3） · 输出 1.5 · 缓存读取 —",
        )
