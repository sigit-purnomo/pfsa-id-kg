from __future__ import annotations

import gc
import hashlib
import html
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from pyvis.network import Network

from pfsa_inference import (
    APP_INFERENCE_SCHEMA_VERSION,
    build_nx_kg,
    dataframe_to_csv_bytes,
    events_to_dataframe,
    events_to_jsonl_bytes,
    filter_events_by_statement_type,
    graph_edges_dataframe,
    graphml_bytes,
    infer_with_disk_cache,
    inspect_artifact_dir,
    list_runtime_devices,
    load_model_bundle,
    normalize_doc_id,
    rdf_bytes,
    relation_rows_for_event,
    release_model_bundle,
    statement_type_counts,
    spans_to_dataframe,
)

APP_TITLE = "PFSA-ID Statement–Event Explorer"
DEFAULT_ARTIFACT_DIR = os.getenv("PFSA_MODEL_ARTIFACT_DIR", str(Path(__file__).resolve().parent / "model_artifacts"))
DEFAULT_CACHE_DIR = os.getenv("PFSA_APP_CACHE_DIR", str(Path(__file__).resolve().parent / "runtime_cache"))
DEFAULT_DEVICE = os.getenv("PFSA_APP_DEVICE", "auto")

APP_DIR = Path(__file__).resolve().parent
SAMPLE_NEWS_PATH = APP_DIR / "sample_news.csv"

FALLBACK_EXAMPLES = {
    "demo_direct_indirect": {
        "doc_id": "demo_direct_indirect",
        "text": (
            "Kepala Pusat Kebijakan Digital Arif Nugraha mengatakan, \"Audit sistem AI harus dapat ditelusuri sampai ke sumber datanya.\" "
            "Ia menegaskan bahwa lembaga publik juga perlu menyimpan bukti keputusan otomatis. "
            "Menurut Arif, dokumentasi yang lengkap penting agar masyarakat dapat memahami dasar sebuah keputusan."
        ),
    },
    "demo_indirect": {
        "doc_id": "demo_indirect",
        "text": (
            "Direktur Pusat Kebijakan Digital Raka Pratama mengatakan pemerintah perlu memperkuat perlindungan data masyarakat. "
            "Ia menambahkan bahwa regulasi baru harus memberikan mekanisme pengaduan yang jelas. "
            "Menurutnya, transparansi penggunaan kecerdasan artifisial juga perlu ditingkatkan."
        ),
    },
}


def _sample_label(row: dict, index: int) -> str:
    title = str(
        row.get("title")
        or row.get("name")
        or row.get("label")
        or row.get("doc_id")
        or f"Sample {index + 1}"
    ).strip()
    preview = re.sub(r"\\s+", " ", str(row.get("text") or "")).strip()
    if len(preview) > 72:
        preview = preview[:69] + "..."
    return f"{title} · {preview}" if preview else title


def load_sample_news(path: Path = SAMPLE_NEWS_PATH) -> Dict[str, dict]:
    """Load sample_news.csv automatically and return display-label -> article mapping."""
    samples: Dict[str, dict] = {}

    if path.exists():
        try:
            df = pd.read_csv(path)
            required = {"doc_id", "text"}
            if required.issubset(df.columns):
                for i, row in enumerate(df.fillna("").to_dict("records")):
                    doc_id = str(row.get("doc_id") or "").strip()
                    article_text = str(row.get("text") or "").strip()
                    if not article_text:
                        continue
                    if not doc_id:
                        doc_id = f"sample_{i + 1:03d}"
                    item = {**row, "doc_id": doc_id, "text": article_text}
                    label = _sample_label(item, i)
                    # Keep labels unique even when titles/doc_ids repeat.
                    if label in samples:
                        label = f"{label} [{i + 1}]"
                    samples[label] = item
        except Exception:
            samples = {}

    if not samples:
        for i, item in enumerate(FALLBACK_EXAMPLES.values()):
            label = _sample_label(item, i)
            samples[label] = dict(item)

    return samples


SAMPLE_NEWS = load_sample_news()



st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    .block-container {max-width: 1320px; padding-top: 1.2rem; padding-bottom: 3rem;}
    .hero {padding: 1rem 0 1.2rem 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;}
    .hero h1 {font-size: 2rem; margin: 0; letter-spacing: -.02em;}
    .hero p {margin: .35rem 0 0 0; color: #64748b; font-size: 1rem;}
    .step {border: 1px solid #e2e8f0; border-radius: 14px; padding: .85rem 1rem; background: #fff; min-height: 82px;}
    .step-num {font-size: .72rem; font-weight: 700; color: #2563eb; text-transform: uppercase; letter-spacing: .06em;}
    .step-title {font-weight: 700; margin-top: .15rem;}
    .step-text {font-size: .84rem; color: #64748b; margin-top: .2rem;}
    .statement-box {border-radius: 10px; padding: .9rem 1rem; font-size: 1.05rem; line-height: 1.65; border-left: 4px solid #64748b; background: #f8fafc;}
    .statement-direct {border-left-color:#2563eb; background:#eff6ff;}
    .statement-indirect {border-left-color:#d97706; background:#fffbeb;}
    .type-badge {display:inline-block; margin:0 .35rem .45rem 0; padding:.18rem .5rem; border-radius:999px; font-size:.75rem; font-weight:700;}
    .type-direct {background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd;}
    .type-indirect {background:#fef3c7; color:#b45309; border:1px solid #fcd34d;}
    .evidence-box {border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.1rem; line-height: 2.05; background: #fff;}
    .ev-statement {background:#dbeafe; border-bottom:2px solid #2563eb; padding:.06rem .1rem; border-radius:3px;}
    .ev-statement-direct {background:#dbeafe; border-bottom:2px solid #2563eb; padding:.06rem .1rem; border-radius:3px;}
    .ev-statement-indirect {background:#fff3cd; border-bottom:2px solid #d97706; padding:.06rem .1rem; border-radius:3px;}
    .ev-cue {background:#fef3c7; border-bottom:2px solid #ca8a04; padding:.06rem .1rem; border-radius:3px;}
    .ev-speaker {background:#dcfce7; border-bottom:2px solid #16a34a; padding:.06rem .1rem; border-radius:3px;}
    .ev-role {background:#f3e8ff; border-bottom:2px solid #9333ea; padding:.06rem .1rem; border-radius:3px;}
    .ev-affiliation {background:#fae8ff; border-bottom:2px solid #c026d3; padding:.06rem .1rem; border-radius:3px;}
    .ev-datetime {background:#cffafe; border-bottom:2px solid #0891b2; padding:.06rem .1rem; border-radius:3px;}
    .ev-location {background:#ffedd5; border-bottom:2px solid #ea580c; padding:.06rem .1rem; border-radius:3px;}
    .ev-event {background:#f1f5f9; border-bottom:2px solid #475569; padding:.06rem .1rem; border-radius:3px;}
    .ev-issue {background:#ffe4e6; border-bottom:2px solid #e11d48; padding:.06rem .1rem; border-radius:3px;}
    .in-direct-statement {outline:1px solid rgba(37,99,235,.42); outline-offset:1px;}
    .in-indirect-statement {outline:1px solid rgba(217,119,6,.48); outline-offset:1px;}
    .ev-speaker {background:#dcfce7; border-bottom:2px solid #16a34a; padding:.06rem .1rem; border-radius:3px;}
    .ev-role {background:#f3e8ff; border-bottom:2px solid #9333ea; padding:.06rem .1rem; border-radius:3px;}
    .ev-affiliation {background:#fae8ff; border-bottom:2px solid #c026d3; padding:.06rem .1rem; border-radius:3px;}
    .ev-datetime {background:#cffafe; border-bottom:2px solid #0891b2; padding:.06rem .1rem; border-radius:3px;}
    .ev-location {background:#ffedd5; border-bottom:2px solid #ea580c; padding:.06rem .1rem; border-radius:3px;}
    .ev-event {background:#f1f5f9; border-bottom:2px solid #475569; padding:.06rem .1rem; border-radius:3px;}
    .ev-issue {background:#ffe4e6; border-bottom:2px solid #e11d48; padding:.06rem .1rem; border-radius:3px;}
    .legend {display:inline-block; margin:.1rem .18rem .4rem 0; padding:.16rem .42rem; border-radius:999px; border:1px solid #d1d5db; font-size:.75rem;}
    .statement-position {font-size:.82rem; color:#64748b; margin-top:.15rem; margin-bottom:.65rem;}
    .view-heading {font-size:.82rem; font-weight:800; color:#475569; text-transform:uppercase; letter-spacing:.06em; margin:.9rem 0 .45rem 0;}
    .nav-help {font-size:.78rem; color:#64748b; margin-top:.2rem;}
    div[data-testid="stMetric"] {border: 1px solid #e2e8f0; border-radius: 12px; padding: .55rem .7rem;}
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {font-weight:800;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    first_sample = next(iter(SAMPLE_NEWS)) if SAMPLE_NEWS else None
    defaults = {
        "bundle": None,
        "bundle_key": None,
        "result": None,
        "article_source_mode": "Sample news",
        "sample_news_selection": first_sample,
        "custom_doc_id": "",
        "custom_news_text": "",
        "statement_type_filter": "All",
        "statement_nav_index": 0,
        "statement_nav_scope": None,
        "detail_view": "Extraction",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def clear_model():
    bundle = st.session_state.get("bundle")
    if bundle is not None:
        release_model_bundle(bundle)
    st.session_state.bundle = None
    st.session_state.bundle_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_label(row):
    if row["device"] == "cpu":
        return "CPU"
    free = "?" if row.get("free_gb") is None else f"{row['free_gb']:.1f} GB free"
    return f"{row['device']} · {row['name']} · {free}"


def event_options(events: List[dict]) -> Dict[str, dict]:
    out = {}
    for i, event in enumerate(events, 1):
        st_text = (event.get("statement") or {}).get("text", "")
        speaker = (event.get("speaker") or {}).get("canonical") or (event.get("speaker") or {}).get("text") or "Unknown speaker"
        short = st_text if len(st_text) <= 88 else st_text[:85] + "..."
        typ = str(event.get("statement_type") or "—").upper()
        marker = "D" if typ == "DIRECT" else "I" if typ == "INDIRECT" else "•"
        out[f"{i}. {marker} · {typ} · {speaker} · {short}"] = event
    return out


def event_spans(event):
    mapping = [
        ("statement", "STATEMENT", "ev-statement"), ("cue", "CUE", "ev-cue"),
        ("speaker", "SPEAKER", "ev-speaker"), ("role", "ROLE", "ev-role"),
        ("affiliation", "AFFILIATION", "ev-affiliation"), ("datetime", "DATETIME", "ev-datetime"),
        ("location", "LOCATION", "ev-location"), ("utterance_event", "EVENT", "ev-event"),
        ("issue", "ISSUE", "ev-issue"),
    ]
    spans = []
    for field, label, css in mapping:
        obj = event.get(field)
        if obj and obj.get("start") is not None and obj.get("end") is not None:
            spans.append({"field": field, "label": label, "css": css, "start": int(obj["start"]), "end": int(obj["end"])})
    return sorted(spans, key=lambda x: (x["start"], -(x["end"] - x["start"])))


def highlighted_text(text, event):
    accepted, cursor_end = [], -1
    for span in event_spans(event):
        if 0 <= span["start"] < span["end"] <= len(text) and span["start"] >= cursor_end:
            accepted.append(span)
            cursor_end = span["end"]
    parts, cursor = [], 0
    for span in accepted:
        parts.append(html.escape(text[cursor:span["start"]]))
        seg = html.escape(text[span["start"]:span["end"]])
        parts.append(f'<span class="{span["css"]}" title="{span["label"]}">{seg}</span>')
        cursor = span["end"]
    parts.append(html.escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def _speaker_evidence_label(obj: dict) -> str:
    label = str(
        obj.get("mention_label")
        or obj.get("label")
        or "PERSON"
    ).upper()
    return label if label in {"PERSON", "PERSONCOREF"} else "PERSON"


def _cue_evidence_label(obj: dict) -> str:
    label = str(obj.get("label") or "CUE").upper()
    return label if label in {"CUE", "CUECOREF"} else "CUE"


def event_evidence_spans(event: dict) -> List[dict]:
    """Return every evidence field for one StatementEvent."""
    statement_type = str(event.get("statement_type") or "").upper()
    event_id = event.get("event_id")
    spans = []

    field_specs = [
        (
            "statement",
            (
                "STATEMENT_DIRECT"
                if statement_type == "DIRECT"
                else "STATEMENT_INDIRECT"
            ),
            (
                "ev-statement-direct"
                if statement_type == "DIRECT"
                else "ev-statement-indirect"
            ),
        ),
        ("cue", None, "ev-cue"),
        ("speaker", None, "ev-speaker"),
        ("role", "ROLE", "ev-role"),
        ("affiliation", "AFFILIATION", "ev-affiliation"),
        ("datetime", "DATETIME", "ev-datetime"),
        ("location", "LOCATION", "ev-location"),
        ("utterance_event", "EVENT", "ev-event"),
        ("issue", "ISSUE", "ev-issue"),
    ]

    for field, fixed_label, css in field_specs:
        obj = event.get(field)
        if not isinstance(obj, dict):
            continue
        if obj.get("start") is None or obj.get("end") is None:
            continue

        start, end = int(obj["start"]), int(obj["end"])
        if start >= end:
            continue

        if field == "speaker":
            label = _speaker_evidence_label(obj)
        elif field == "cue":
            label = _cue_evidence_label(obj)
        else:
            label = fixed_label

        spans.append({
            "event_id": event_id,
            "statement_type": statement_type,
            "field": field,
            "label": label,
            "css": css,
            "start": start,
            "end": end,
        })

    return spans


def evidence_spans_for_events(events: List[dict]) -> List[dict]:
    """Collect all evidence labels for the visible StatementEvents."""
    merged = {}

    for event in events:
        for span in event_evidence_spans(event):
            key = (
                span["start"],
                span["end"],
                span["field"],
                span["label"],
                span["statement_type"],
            )
            if key not in merged:
                merged[key] = {
                    **span,
                    "event_ids": [span.get("event_id")],
                }
            else:
                eid = span.get("event_id")
                if eid not in merged[key]["event_ids"]:
                    merged[key]["event_ids"].append(eid)

    return sorted(
        merged.values(),
        key=lambda s: (
            s["start"],
            -(s["end"] - s["start"]),
            s["field"],
            s["label"],
        ),
    )


EVIDENCE_FIELD_PRIORITY = {
    # More specific labels override the STATEMENT background when they overlap.
    "cue": 90,
    "speaker": 85,
    "role": 80,
    "affiliation": 75,
    "datetime": 70,
    "location": 65,
    "utterance_event": 60,
    "issue": 55,
    "statement": 10,
}


def highlighted_evidence_text(text: str, events: List[dict]) -> str:
    """Render all evidence labels without allowing STATEMENT to hide other labels.

    The article is segmented at every evidence boundary. For each segment, the
    most specific active evidence label controls the background color. If that
    segment also belongs to a DIRECT/INDIRECT statement, a thin outline keeps
    the statement membership visible.
    """
    spans = [
        s
        for s in evidence_spans_for_events(events)
        if 0 <= s["start"] < s["end"] <= len(text)
    ]
    if not spans:
        return html.escape(text).replace("\n", "<br>")

    boundaries = {0, len(text)}
    for span in spans:
        boundaries.add(span["start"])
        boundaries.add(span["end"])
    boundaries = sorted(boundaries)

    parts = []

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if left >= right:
            continue

        raw_segment = text[left:right]
        segment = html.escape(raw_segment)

        active = [
            s
            for s in spans
            if s["start"] <= left and right <= s["end"]
        ]
        if not active:
            parts.append(segment)
            continue

        statement_spans = [
            s for s in active
            if s["field"] == "statement"
        ]
        specific_spans = [
            s for s in active
            if s["field"] != "statement"
        ]

        if specific_spans:
            primary = max(
                specific_spans,
                key=lambda s: (
                    EVIDENCE_FIELD_PRIORITY.get(s["field"], 0),
                    -(s["end"] - s["start"]),
                    s["label"],
                ),
            )
        else:
            primary = max(
                statement_spans,
                key=lambda s: (
                    1 if s["statement_type"] == "DIRECT" else 0,
                    -(s["end"] - s["start"]),
                ),
            )

        classes = [primary["css"]]

        # Preserve statement membership when a more-specific label occupies the
        # same character interval.
        statement_types = {
            s["statement_type"]
            for s in statement_spans
        }
        if specific_spans:
            if "DIRECT" in statement_types:
                classes.append("in-direct-statement")
            elif "INDIRECT" in statement_types:
                classes.append("in-indirect-statement")

        labels = []
        for s in sorted(
            active,
            key=lambda x: (
                -EVIDENCE_FIELD_PRIORITY.get(x["field"], 0),
                x["label"],
            ),
        ):
            tag = s["label"]
            if tag not in labels:
                labels.append(tag)

        parts.append(
            f'<span class="{" ".join(classes)}" '
            f'title="{html.escape(" + ".join(labels), quote=True)}">'
            f'{segment}</span>'
        )

    return "".join(parts).replace("\n", "<br>")


def evidence_spans_dataframe(events: List[dict]) -> pd.DataFrame:
    rows = []
    for s in evidence_spans_for_events(events):
        rows.append({
            "event_ids": ", ".join(
                str(x)
                for x in s.get("event_ids", [])
                if x is not None
            ),
            "statement_type": s.get("statement_type"),
            "field": s.get("field"),
            "label": s.get("label"),
            "start": s.get("start"),
            "end": s.get("end"),
        })
    return pd.DataFrame(rows)


def render_evidence_legend(type_filter: str, events: List[dict]):
    """Show only evidence labels that are present in the current filter."""
    labels_present = {
        s["label"]
        for s in evidence_spans_for_events(events)
    }

    chips = []

    if "STATEMENT_DIRECT" in labels_present:
        chips.append(
            '<span class="legend ev-statement-direct">STATEMENT · DIRECT</span>'
        )
    if "STATEMENT_INDIRECT" in labels_present:
        chips.append(
            '<span class="legend ev-statement-indirect">STATEMENT · INDIRECT</span>'
        )

    legend_specs = [
        ({"CUE", "CUECOREF"}, "CUE / CUECOREF", "ev-cue"),
        ({"PERSON", "PERSONCOREF"}, "PERSON / PERSONCOREF", "ev-speaker"),
        ({"ROLE"}, "ROLE", "ev-role"),
        ({"AFFILIATION"}, "AFFILIATION", "ev-affiliation"),
        ({"DATETIME"}, "DATETIME", "ev-datetime"),
        ({"LOCATION"}, "LOCATION", "ev-location"),
        ({"EVENT"}, "EVENT", "ev-event"),
        ({"ISSUE"}, "ISSUE", "ev-issue"),
    ]

    for labels, title, css in legend_specs:
        if labels_present & labels:
            chips.append(
                f'<span class="legend {css}">{title}</span>'
            )

    st.markdown(
        "".join(chips),
        unsafe_allow_html=True,
    )


def render_legend():
    st.markdown(
        '<span class="legend ev-statement">Statement</span><span class="legend ev-cue">Cue</span>'
        '<span class="legend ev-speaker">Speaker</span><span class="legend ev-role">Role</span>'
        '<span class="legend ev-affiliation">Affiliation</span><span class="legend ev-datetime">Date/time</span>'
        '<span class="legend ev-location">Location</span><span class="legend ev-event">Event</span>'
        '<span class="legend ev-issue">Issue</span>',
        unsafe_allow_html=True,
    )


def graph_html(graph: nx.MultiDiGraph, height=610):
    if graph.number_of_nodes() == 0:
        return "<div style='padding:1rem'>No graph nodes.</div>"

    net = Network(
        height=f"{height}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#0f172a",
        cdn_resources="in_line",
    )

    entity_colors = {
        "Article": "#e2e8f0",
        "Person": "#bbf7d0",
        "Role": "#e9d5ff",
        "Organization": "#f5d0fe",
        "DateTime": "#a5f3fc",
        "Location": "#fed7aa",
        "UtteranceEvent": "#cbd5e1",
        "Issue": "#fecdd3",
    }

    for node_id, attrs in graph.nodes(data=True):
        typ = attrs.get("node_type", "Entity")
        label = str(attrs.get("label") or attrs.get("content") or node_id)
        shape = "ellipse"
        color = entity_colors.get(typ, "#e5e7eb")
        size = 20

        if typ == "Article":
            shape = "box"
        elif typ == "StatementEvent":
            stype = str(attrs.get("statement_type") or "").upper()
            shape = "dot"
            size = 30
            if stype == "DIRECT":
                color = "#93c5fd"
                prefix = "DIRECT"
            elif stype == "INDIRECT":
                color = "#fcd34d"
                prefix = "INDIRECT"
            else:
                color = "#cbd5e1"
                prefix = "STATEMENT"
            if len(label) > 64:
                label = label[:61] + "..."
            label = f"{prefix}\n{label}"

        title = "<br>".join(
            f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}"
            for k, v in attrs.items() if v not in (None, "")
        )
        net.add_node(
            node_id,
            label=label,
            title=title,
            shape=shape,
            color=color,
            size=size,
        )

    for source, target, _key, attrs in graph.edges(keys=True, data=True):
        rel = attrs.get("relation", "")
        conf = attrs.get("relation_confidence")
        label = rel if conf is None else f"{rel} ({float(conf):.2f})"
        source_attrs = graph.nodes[source] if source in graph.nodes else {}
        stype = str(source_attrs.get("statement_type") or "").upper()
        edge_color = "#2563eb" if stype == "DIRECT" else "#d97706" if stype == "INDIRECT" else "#64748b"
        net.add_edge(source, target, label=label, arrows="to", color=edge_color)

    net.barnes_hut(
        gravity=-3800,
        central_gravity=.18,
        spring_length=150,
        spring_strength=.045,
        damping=.9,
    )
    return net.generate_html(notebook=False)


def event_detail(event):
    statement = event.get("statement") or {}
    statement_type = str(event.get("statement_type") or "UNKNOWN").upper()
    badge_class = "type-direct" if statement_type == "DIRECT" else "type-indirect" if statement_type == "INDIRECT" else ""
    box_class = "statement-direct" if statement_type == "DIRECT" else "statement-indirect" if statement_type == "INDIRECT" else ""
    st.markdown(
        f'<span class="type-badge {badge_class}">{html.escape(statement_type)}</span>'
        f'<div class="statement-box {box_class}">{html.escape(statement.get("text") or "—")}</div>',
        unsafe_allow_html=True,
    )
    rows = [
        ("Type", event.get("statement_type"), None),
        ("Cue", (event.get("cue") or {}).get("text"), (event.get("cue") or {}).get("label")),
        ("Speaker", (event.get("speaker") or {}).get("canonical") or (event.get("speaker") or {}).get("text"), (event.get("speaker") or {}).get("mention_label")),
        ("Role", (event.get("role") or {}).get("text"), None),
        ("Affiliation", (event.get("affiliation") or {}).get("text"), None),
        ("Date/time", (event.get("datetime") or {}).get("text"), None),
        ("Location", (event.get("location") or {}).get("text"), None),
        ("Utterance event", (event.get("utterance_event") or {}).get("text"), None),
        ("Issue", (event.get("issue") or {}).get("text"), "sentence-level"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Element", "Value", "Label"]), hide_index=True, use_container_width=True)
    rel = pd.DataFrame(relation_rows_for_event(event))
    if not rel.empty:
        with st.expander("Relation confidence"):
            st.dataframe(rel, hide_index=True, use_container_width=True)


st.markdown(f'<div class="hero"><h1>{APP_TITLE}</h1><p>Extract direct and indirect public-figure statements, inspect their evidence, and explore the resulting knowledge graph.</p></div>', unsafe_allow_html=True)

steps = st.columns(3)
steps[0].markdown('<div class="step"><div class="step-num">Step 1</div><div class="step-title">Enter news article</div><div class="step-text">Choose an article from sample_news.csv or paste your own Indonesian news text.</div></div>', unsafe_allow_html=True)
steps[1].markdown('<div class="step"><div class="step-num">Step 2</div><div class="step-title">Run inference</div><div class="step-text">Joint DIRECT + INDIRECT BILUO-CRF decoding, ISSUE sentence selection, and relations.</div></div>', unsafe_allow_html=True)
steps[2].markdown('<div class="step"><div class="step-num">Step 3</div><div class="step-title">Explore the KG</div><div class="step-text">Compare DIRECT and INDIRECT StatementEvents and their graph relations.</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Model")
    artifact_dir = st.text_input("Model artifact", DEFAULT_ARTIFACT_DIR, label_visibility="collapsed")
    audit = inspect_artifact_dir(Path(artifact_dir), verify_checkpoint=False)
    if audit["ready"]:
        st.success("Model artifact ready")
    else:
        st.warning("Model artifact not ready")
        for p in audit["problems"]:
            st.caption("• " + p)

    devices = list_runtime_devices()
    device_values = ["auto"] + [r["device"] for r in devices]
    device = st.selectbox("Device", device_values, index=device_values.index(DEFAULT_DEVICE) if DEFAULT_DEVICE in device_values else 0,
                          format_func=lambda x: "Auto" if x == "auto" else device_label(next(r for r in devices if r["device"] == x)))

    model_key = (str(Path(artifact_dir).resolve()), device)
    loaded = st.session_state.get("bundle") is not None and st.session_state.get("bundle_key") == model_key
    if loaded:
        st.success(f"Loaded on {st.session_state.bundle.device}")
    elif st.session_state.get("bundle") is not None:
        st.info("Model settings changed. Reload model.")

    if st.button("Load model" if not loaded else "Reload model", type="primary", use_container_width=True, disabled=not audit["ready"]):
        clear_model()
        try:
            with st.spinner("Loading student model..."):
                st.session_state.bundle = load_model_bundle(Path(artifact_dir), relation_threshold=.50, device=device)
                st.session_state.bundle_key = model_key
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.get("bundle") is not None and st.button("Release model", use_container_width=True):
        clear_model()
        st.rerun()

    with st.expander("Advanced settings"):
        relation_threshold = st.slider("Relation threshold", .05, .95, .50, .05)
        use_cache = st.checkbox("Reuse inference cache", True)
        cache_dir = st.text_input("Cache directory", DEFAULT_CACHE_DIR)
        if loaded:
            b = st.session_state.bundle
            st.caption(f"Student: {b.model_id}")
            teacher = b.bundle_manifest.get("teacher_model_id")
            if teacher:
                st.caption(f"Training teacher: {teacher}")
            weights = b.bundle_manifest.get("model_selection_weights") or {}
            if weights.get("direct") is not None and weights.get("indirect") is not None:
                st.caption(
                    "Checkpoint selection: "
                    f"DIRECT {float(weights['direct']):.0%} · "
                    f"INDIRECT {float(weights['indirect']):.0%}"
                )
            st.caption(f"Inference schema: {APP_INFERENCE_SCHEMA_VERSION}")
            st.caption(f"Checkpoint: {b.checkpoint_sha256[:12]}…")

bundle_ready = st.session_state.get("bundle") is not None and st.session_state.get("bundle_key") == model_key

st.subheader("1. News article")

source_mode = st.radio(
    "Article source",
    ["Sample news", "Paste text"],
    horizontal=True,
    key="article_source_mode",
)

if source_mode == "Sample news":
    sample_labels = list(SAMPLE_NEWS)
    if not sample_labels:
        st.error("No valid sample article is available.")
        doc_id, text = "", ""
    else:
        if st.session_state.get("sample_news_selection") not in SAMPLE_NEWS:
            st.session_state.sample_news_selection = sample_labels[0]

        selected_sample_label = st.selectbox(
            "Sample article",
            sample_labels,
            key="sample_news_selection",
        )
        selected_sample = SAMPLE_NEWS[selected_sample_label]
        doc_id = str(selected_sample.get("doc_id") or "").strip()
        text = str(selected_sample.get("text") or "")

        source_name = "sample_news.csv" if SAMPLE_NEWS_PATH.exists() else "built-in fallback samples"
        st.caption(f"Loaded automatically from {source_name} · {len(SAMPLE_NEWS)} sample article(s) available.")

        sample_meta, sample_id = st.columns([3, 1])
        sample_meta.text_area(
            "Article text",
            value=text,
            height=250,
            disabled=True,
            help="Switch to Paste text if you want to enter or edit another article.",
        )
        sample_id.text_input(
            "Document ID",
            value=doc_id,
            disabled=True,
        )
else:
    custom_id_col, _ = st.columns([1, 2])
    doc_id = custom_id_col.text_input(
        "Document ID (optional)",
        key="custom_doc_id",
        placeholder="Leave blank to generate one automatically",
    )
    text = st.text_area(
        "Article text",
        key="custom_news_text",
        height=250,
        placeholder="Paste Indonesian news article here...",
    )

current_input_signature = hashlib.sha1(
    f"{source_mode}|{doc_id}|{text}".encode("utf-8")
).hexdigest()

analyze = st.button(
    "Analyze article",
    type="primary",
    use_container_width=True,
    disabled=not bundle_ready or len(text.strip()) < 20,
)
if not bundle_ready:
    st.caption("Load a compatible model artifact from the sidebar before running inference.")

if analyze:
    try:
        started = time.perf_counter()
        final_doc_id = normalize_doc_id(doc_id, text)
        with st.spinner("Extracting StatementEvents and building knowledge graph..."):
            events, spans, cache_hit = infer_with_disk_cache(
                st.session_state.bundle, text, final_doc_id, Path(cache_dir), relation_threshold, use_cache
            )
        st.session_state.result = {
            "events": events,
            "spans": spans,
            "text": text,
            "doc_id": final_doc_id,
            "elapsed": time.perf_counter() - started,
            "cache_hit": cache_hit,
            "input_signature": current_input_signature,
        }
    except Exception as exc:
        st.error(f"Inference failed: {exc}")

result = st.session_state.get("result")
if result is not None and result.get("input_signature") != current_input_signature:
    st.info("Article input changed. Click Analyze article to refresh the extraction result.")
    result = None

if result is not None:
    st.divider()
    st.subheader("2. Extraction result")
    events = result["events"]
    df = events_to_dataframe(events)
    counts = statement_type_counts(events)
    speakers = int(df["speaker"].dropna().nunique()) if not df.empty else 0

    m = st.columns(5)
    m[0].metric("Statements", counts["ALL"])
    m[1].metric("Direct", counts["DIRECT"])
    m[2].metric("Indirect", counts["INDIRECT"])
    m[3].metric("Speakers", speakers)
    m[4].metric("Inference", f"{result['elapsed']:.2f}s")

    if not events:
        st.info("No direct or indirect StatementEvent was produced for this article.")
    else:
        st.markdown("#### Statement type")
        type_filter = st.radio(
            "Filter extracted statements",
            ["All", "Direct", "Indirect"],
            horizontal=True,
            label_visibility="collapsed",
            key="statement_type_filter",
        )
        filtered_events = filter_events_by_statement_type(events, type_filter)
        filtered_df = events_to_dataframe(filtered_events)

        st.caption(
            f"Showing {len(filtered_events)} of {len(events)} extracted StatementEvents. "
            "The filter changes only the display; inference always decodes both DIRECT and INDIRECT labels."
        )

        if not filtered_events:
            st.info(f"No {type_filter.lower()} statements were detected in this article.")
        else:
            options = event_options(filtered_events)
            option_labels = list(options.keys())

            # The navigator scope changes whenever the visible statement set changes.
            # This avoids carrying a stale selection across filters or new inference results.
            event_signature = "|".join(
                str(e.get("event_id") or f"{e.get('statement_type')}:{(e.get('statement') or {}).get('start')}:{(e.get('statement') or {}).get('end')}")
                for e in filtered_events
            )
            nav_scope = hashlib.sha1(
                f"{result['doc_id']}|{type_filter}|{event_signature}".encode("utf-8")
            ).hexdigest()[:12]
            selector_key = f"statement_selector_{nav_scope}"

            if st.session_state.get("statement_nav_scope") != nav_scope:
                st.session_state.statement_nav_scope = nav_scope
                st.session_state.statement_nav_index = 0
                st.session_state[selector_key] = option_labels[0]

            current_index = int(st.session_state.get("statement_nav_index", 0))
            current_index = max(0, min(current_index, len(option_labels) - 1))

            st.markdown("#### Browse extracted statements")
            nav_prev, nav_select, nav_next = st.columns([1.15, 7.7, 1.15])
            nav_prev.markdown('<div style="height:1.65rem"></div>', unsafe_allow_html=True)
            nav_next.markdown('<div style="height:1.65rem"></div>', unsafe_allow_html=True)

            prev_clicked = nav_prev.button(
                "← Previous",
                use_container_width=True,
                disabled=current_index <= 0,
                key=f"prev_statement_{nav_scope}",
            )
            next_clicked = nav_next.button(
                "Next →",
                use_container_width=True,
                disabled=current_index >= len(option_labels) - 1,
                key=f"next_statement_{nav_scope}",
            )

            if prev_clicked or next_clicked:
                current_index += -1 if prev_clicked else 1
                current_index = max(0, min(current_index, len(option_labels) - 1))
                st.session_state.statement_nav_index = current_index
                st.session_state[selector_key] = option_labels[current_index]
                st.rerun()

            selected_label = nav_select.selectbox(
                "Statement to inspect",
                option_labels,
                index=current_index,
                key=selector_key,
            )
            selected_index = option_labels.index(selected_label)
            st.session_state.statement_nav_index = selected_index
            selected_event = options[selected_label]

            selected_type = str(selected_event.get("statement_type") or "UNKNOWN").upper()
            st.markdown(
                f'<div class="statement-position">Statement {selected_index + 1} of {len(filtered_events)} · '
                f'{html.escape(selected_type)} · use Previous/Next to browse sequentially.</div>',
                unsafe_allow_html=True,
            )

            # A persistent three-button view menu is clearer than subtle native tabs and
            # preserves the active view when Previous/Next triggers a rerun.
            st.markdown('<div class="view-heading">View details</div>', unsafe_allow_html=True)
            view_cols = st.columns(3)
            view_specs = [
                ("Extraction", "📋 Extraction"),
                ("Evidence", "🔎 Evidence"),
                ("Knowledge Graph", "🕸 Knowledge Graph"),
            ]
            active_view = st.session_state.get("detail_view", "Extraction")
            for col, (view_name, button_label) in zip(view_cols, view_specs):
                if col.button(
                    button_label,
                    use_container_width=True,
                    type="primary" if active_view == view_name else "secondary",
                    key=f"view_{view_name.replace(' ', '_').lower()}",
                ):
                    if active_view != view_name:
                        st.session_state.detail_view = view_name
                        st.rerun()

            active_view = st.session_state.get("detail_view", "Extraction")

            if active_view == "Extraction":
                event_detail(selected_event)
                with st.expander("Statements in current filter", expanded=False):
                    cols = [
                        "statement_type", "statement", "statement_confidence", "cue", "cue_label",
                        "speaker", "speaker_mention_label", "role", "affiliation", "datetime",
                        "location", "utterance_event", "issue",
                    ]
                    st.dataframe(
                        filtered_df[[c for c in cols if c in filtered_df.columns]],
                        hide_index=True,
                        use_container_width=True,
                    )

            elif active_view == "Evidence":
                # Evidence is article-level and follows the global Statement type filter.
                evidence_events = filtered_events

                direct_visible = sum(
                    str(e.get("statement_type") or "").upper() == "DIRECT"
                    for e in evidence_events
                )
                indirect_visible = sum(
                    str(e.get("statement_type") or "").upper() == "INDIRECT"
                    for e in evidence_events
                )

                if type_filter == "All":
                    st.caption(
                        f"All evidence: {direct_visible} DIRECT and "
                        f"{indirect_visible} INDIRECT StatementEvent(s), including "
                        "STATEMENT, CUE, PERSON, ROLE, AFFILIATION, DATETIME, LOCATION, EVENT, and ISSUE where available."
                    )
                elif type_filter == "Direct":
                    st.caption(
                        f"DIRECT evidence only: {direct_visible} StatementEvent(s), with all available evidence labels."
                    )
                else:
                    st.caption(
                        f"INDIRECT evidence only: {indirect_visible} StatementEvent(s), with all available evidence labels."
                    )

                render_evidence_legend(type_filter, evidence_events)

                st.markdown(
                    f'<div class="evidence-box">'
                    f'{highlighted_evidence_text(result["text"], evidence_events)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("Evidence spans in current filter"):
                    evidence_df = evidence_spans_dataframe(evidence_events)
                    st.dataframe(
                        evidence_df,
                        hide_index=True,
                        use_container_width=True,
                    )

                with st.expander("Selected StatementEvent JSON"):
                    st.json(selected_event)

            else:
                st.markdown(
                    '<span class="type-badge type-direct">DIRECT StatementEvent</span>'
                    '<span class="type-badge type-indirect">INDIRECT StatementEvent</span>',
                    unsafe_allow_html=True,
                )
                graph_scope = st.radio(
                    "Graph scope",
                    ["All extracted statements", "Current type filter", "Selected statement"],
                    horizontal=True,
                )
                if graph_scope == "Selected statement":
                    graph_events = [selected_event]
                elif graph_scope == "Current type filter":
                    graph_events = filtered_events
                else:
                    graph_events = events

                graph = build_nx_kg(graph_events)
                graph_counts = statement_type_counts(graph_events)
                gm = st.columns(5)
                gm[0].metric("Nodes", graph.number_of_nodes())
                gm[1].metric("Edges", graph.number_of_edges())
                gm[2].metric("Statements", graph_counts["ALL"])
                gm[3].metric("Direct", graph_counts["DIRECT"])
                gm[4].metric("Indirect", graph_counts["INDIRECT"])
                components.html(graph_html(graph), height=630, scrolling=True)
                st.caption(
                    "Blue StatementEvent nodes are DIRECT; amber StatementEvent nodes are INDIRECT. "
                    "Outgoing relation edges follow the same type color."
                )
                with st.expander("Graph edges"):
                    st.dataframe(graph_edges_dataframe(graph), hide_index=True, use_container_width=True)

        with st.expander("Download results"):
            graph = build_nx_kg(events)
            d = st.columns(4)
            d[0].download_button(
                "Events JSONL", events_to_jsonl_bytes(events),
                f"{result['doc_id']}_events.jsonl", "application/x-ndjson",
                use_container_width=True,
            )
            d[1].download_button(
                "Events CSV", dataframe_to_csv_bytes(df),
                f"{result['doc_id']}_events.csv", "text/csv",
                use_container_width=True,
            )
            d[2].download_button(
                "GraphML", graphml_bytes(graph),
                f"{result['doc_id']}_kg.graphml", "application/xml",
                use_container_width=True,
            )
            d[3].download_button(
                "RDF/Turtle", rdf_bytes(events, "turtle"),
                f"{result['doc_id']}_kg.ttl", "text/turtle",
                use_container_width=True,
            )
            st.caption(
                "Downloads contain both DIRECT and INDIRECT student-model predictions. "
                "The deployment app does not call the training-time Qwen teacher."
            )
