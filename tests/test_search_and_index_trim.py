"""Tests for search_wiki_pages (agent/tools.py) and the index.md context trim
(agent/context_trim.py)."""

from __future__ import annotations

import json

from openkb.agent.context_trim import INDEX_PLACEHOLDER, drop_used_index_reads, index_trim_filter
from openkb.agent.query import build_run_config_from_bundle
from openkb.agent.tools import search_wiki_pages


def _wiki(tmp_path):
    (tmp_path / "entities").mkdir()
    (tmp_path / "summaries").mkdir()
    (tmp_path / "sources").mkdir()
    (tmp_path / "index.md").write_text(
        "# Index\n"
        "- [[entities/tethys]] (organization) — Bettencourt family office PE arm.\n"
        "- [[entities/creadev]] (organization) — Industrial investment arm of AFM.\n"
        "- [[summaries/afir]] (short) — Sogefir intro call.\n",
        encoding="utf-8",
    )
    (tmp_path / "entities" / "tethys.md").write_text(
        "---\nsources: [summaries/hermes-notes.md]\n---\n"
        "# Tethys\nThe Bettencourt family office.\n",
        encoding="utf-8",
    )
    (tmp_path / "entities" / "creadev.md").write_text(
        "# Creadev\nInvestment arm of the Mulliez family offices ecosystem.\n", encoding="utf-8"
    )
    (tmp_path / "summaries" / "afir.md").write_text(
        "# AFIR\nPierre is in a club of family offices (Béthencourt, Hermès).\n", encoding="utf-8"
    )
    (tmp_path / "sources" / "emmi.md").write_text(
        "Beaucoup plus rapide que Renault, Peugeot…\nDaniel said hi.\n", encoding="utf-8"
    )
    return str(tmp_path)


class TestSearchWikiPages:
    def test_accent_insensitive_both_ways(self, tmp_path):
        root = _wiki(tmp_path)
        assert "summaries/afir.md" in search_wiki_pages(["hermes"], root)
        assert "summaries/afir.md" in search_wiki_pages(["Hérmes"], root)

    def test_snippet_keeps_original_accents(self, tmp_path):
        out = search_wiki_pages(["hermes"], _wiki(tmp_path))
        assert "Hermès" in out

    def test_finds_passing_mention_in_sources(self, tmp_path):
        out = search_wiki_pages(["Peugeot"], _wiki(tmp_path))
        assert "sources/emmi.md" in out

    def test_whole_words_only(self, tmp_path):
        # "Niel" must not match "Daniel".
        assert search_wiki_pages(["Niel"], _wiki(tmp_path)).startswith('## "Niel": no match')

    def test_plural_matches(self, tmp_path):
        out = search_wiki_pages(["family office"], _wiki(tmp_path))
        assert "entities/creadev.md" in out  # "family offices" in the body

    def test_frontmatter_not_searched(self, tmp_path):
        # tethys.md only mentions "hermes" in its frontmatter sources list.
        out = search_wiki_pages(["hermes"], _wiki(tmp_path))
        assert "entities/tethys.md" not in out

    def test_description_matches_rank_first(self, tmp_path):
        out = search_wiki_pages(["family office"], _wiki(tmp_path))
        assert out.index("entities/tethys.md") < out.index("entities/creadev.md")

    def test_includes_index_one_liner(self, tmp_path):
        out = search_wiki_pages(["Mulliez"], _wiki(tmp_path))
        assert "entities/creadev.md — (organization) — Industrial investment arm of AFM." in out

    def test_page_types_filter(self, tmp_path):
        out = search_wiki_pages(["family office"], _wiki(tmp_path), page_types=["entities"])
        assert "entities/creadev.md" in out
        assert "summaries/afir.md" not in out

    def test_cap_reports_total(self, tmp_path):
        out = search_wiki_pages(["family office"], _wiki(tmp_path), max_results=1)
        assert '"family office": 3 page(s), showing 1' in out

    def test_reflects_file_changes(self, tmp_path):
        root = _wiki(tmp_path)
        search_wiki_pages(["Drahi"], root)
        (tmp_path / "entities" / "creadev.md").write_text(
            "Patrick Drahi visited.\n", encoding="utf-8"
        )
        assert "entities/creadev.md" in search_wiki_pages(["Drahi"], root)

    def test_footer_states_limits(self, tmp_path):
        assert "Keyword matches only" in search_wiki_pages(["x"], _wiki(tmp_path))


def _call(call_id, path):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "read_file",
        "arguments": json.dumps({"path": path}),
    }


def _output(call_id, text):
    return {"type": "function_call_output", "call_id": call_id, "output": text}


class TestDropUsedIndexReads:
    def test_keeps_index_the_model_has_not_seen_yet(self):
        items = [{"role": "user", "content": "q"}, _call("a", "index.md"), _output("a", "INDEX")]
        assert drop_used_index_reads(items) == items

    def test_keeps_index_read_in_parallel_with_other_reads(self):
        items = [
            {"role": "user", "content": "q"},
            _call("a", "index.md"),
            _call("b", "log.md"),
            _output("a", "INDEX"),
            _output("b", "LOG"),
        ]
        assert drop_used_index_reads(items)[3]["output"] == "INDEX"

    def test_replaces_index_once_the_model_has_responded(self):
        items = [
            {"role": "user", "content": "q"},
            _call("a", "index.md"),
            _output("a", "INDEX"),
            _call("b", "entities/x.md"),
            _output("b", "PAGE"),
        ]
        trimmed = drop_used_index_reads(items)
        assert trimmed[2]["output"] == INDEX_PLACEHOLDER
        assert trimmed[4]["output"] == "PAGE"
        assert items[2]["output"] == "INDEX"  # input not mutated

    def test_assistant_message_counts_as_a_response(self):
        items = [
            {"role": "user", "content": "q"},
            _call("a", "./index.md"),
            _output("a", "INDEX"),
            {"type": "message", "role": "assistant", "content": "notes"},
        ]
        assert drop_used_index_reads(items)[2]["output"] == INDEX_PLACEHOLDER

    def test_other_pages_untouched(self):
        items = [_call("a", "entities/index.md"), _output("a", "PAGE"), _call("b", "x.md")]
        assert drop_used_index_reads(items) == items

    def test_filter_returns_model_input(self):
        from agents.run_config import CallModelData, ModelInputData

        items = [_call("a", "index.md"), _output("a", "INDEX"), _call("b", "x.md")]
        data = CallModelData(
            model_data=ModelInputData(input=items, instructions="sys"), agent=None, context=None
        )
        result = index_trim_filter(data)
        assert result.instructions == "sys"
        assert result.input[1]["output"] == INDEX_PLACEHOLDER


def test_run_config_installs_index_filter():
    from openkb.config import LlmCredentialBundle

    bundle = LlmCredentialBundle(api_key="k", base_url=None)
    run_config = build_run_config_from_bundle("openai/gpt-4o-mini", bundle)
    assert run_config.call_model_input_filter is index_trim_filter
