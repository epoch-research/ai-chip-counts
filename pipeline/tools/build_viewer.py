#!/usr/bin/env python3.11
"""Write standalone copies of the hub explorers that open straight from disk.

    python3.11 -m pipeline.tools.build_viewer

Writes staging/viewer-sales.html and staging/viewer-owners.html. Each is the matching
hub/ page with its stylesheet, the vendored d3, the explorer scripts and every staging
table inlined, so it needs no server, no internet and no other files. The graph and
table views are the hub's own code, so the two never drift apart.
pipeline.build_staging regenerates both on every run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import chip_schema as cs  # noqa: E402

HUB = cs.REPO / "hub"
STAGING = cs.REPO / "staging"
PAGES = {"ai-chip-sales.html": ("viewer-sales.html", "sales.js"),
         "ai-chip-owners.html": ("viewer-owners.html", "owners.js")}
TABLES = cs.TABLES + ["chip_types", "organizations"]


def inline_script(code):
    # "</" inside a script block would end it early.
    return "<script>\n" + code.replace("</", "<\\/") + "\n</script>"


def js_string(text):
    return json.dumps(text).replace("</", "<\\/")


def build():
    data = {t: (STAGING / f"{t}.csv").read_text() for t in TABLES if (STAGING / f"{t}.csv").exists()}
    data["chip_type_map"] = cs.CHIP_TYPE_MAP_PATH.read_text()
    payload = "window.HUB_DATA = {" + ",".join(f"{js_string(k)}:{js_string(v)}" for k, v in data.items()) + "};"
    css = (HUB / "hub.css").read_text()
    d3 = (HUB / "vendor" / "d3.min.js").read_text()
    core = (HUB / "core.js").read_text()
    for page, (out_name, script) in PAGES.items():
        html = (HUB / page).read_text()
        replacements = {
            '<link rel="stylesheet" href="hub.css">': f"<style>\n{css}\n</style>",
            '<script src="vendor/d3.min.js"></script>': inline_script(d3),
            '<script src="core.js"></script>': inline_script(payload) + "\n" + inline_script(core),
            f'<script src="{script}"></script>': inline_script((HUB / script).read_text()),
            'href="index.html"': 'href="viewer-sales.html"',
            'href="ai-chip-sales.html"': 'href="viewer-sales.html"',
            'href="ai-chip-owners.html"': 'href="viewer-owners.html"',
            "Reads the canonical tables in <code>staging/</code>": "A standalone copy with the tables in <code>staging/</code> built in",
        }
        for old, new in replacements.items():
            if old not in html:
                raise SystemExit(f"{page}: expected {old!r} in the page shell")
            html = html.replace(old, new)
        out = STAGING / out_name
        out.write_text(html)
        print(f"wrote {out.relative_to(cs.REPO)} ({out.stat().st_size / 1024:.0f} KB)")
    stale = STAGING / "viewer.html"
    if stale.exists():
        stale.unlink()


if __name__ == "__main__":
    build()
