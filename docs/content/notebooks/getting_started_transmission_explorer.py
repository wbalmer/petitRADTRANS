"""Notebook helper for rendering the getting-started transmission explorer inline."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from IPython.display import HTML, display


NOTEBOOKS_DIR = Path(__file__).resolve().parent
DOCS_DIR = NOTEBOOKS_DIR.parents[1]
HTML_TEMPLATE_PATH = DOCS_DIR / "_static" / "getting_started_transmission_explorer.html"
DATA_PATH = DOCS_DIR / "_static" / "getting_started_transmission_explorer_data.json"


def _build_iframe_srcdoc() -> str:
    html_template = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    explorer_data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    embedded_data_tag = (
        "<script id=\"getting-started-explorer-data\" type=\"application/json\">"
        f"{json.dumps(explorer_data, separators=(',', ':'))}"
        "</script>"
    )

    return html_template.replace(
        "<script id=\"getting-started-explorer-data\" type=\"application/json\"></script>",
        embedded_data_tag,
        1,
    )


def display_getting_started_transmission_explorer(height: int = 632) -> None:
    srcdoc = escape(_build_iframe_srcdoc(), quote=True)
    iframe_html = (
        "<div style=\"width: 100%; overflow: visible;\">"
        f"<iframe srcdoc=\"{srcdoc}\" "
        "title=\"Interactive petitRADTRANS transmission explorer\" "
        "loading=\"lazy\" scrolling=\"no\" "
        f"style=\"width: 100%; height: {height}px; border: 0; border-radius: 18px; overflow: hidden; background: transparent;\"></iframe>"
        "</div>"
    )
    display(HTML(iframe_html))