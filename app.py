from __future__ import annotations

import gc
import html
import io
import json
import os
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
    spans_to_dataframe,
)


APP_TITLE = "PFSA-ID Statement–Event Explorer"
APP_SUBTITLE = (
    "Live document-level inference and evidence-grounded knowledge-graph visualization "
    "for Indonesian public-figure statements."
)

DEFAULT_ARTIFACT_DIR = os.getenv(
    "PFSA_MODEL_ARTIFACT_DIR",
    str(Path(__file__).resolve().parent / "model_artifacts"),
)
DEFAULT_CACHE_DIR = os.getenv(
    "PFSA_APP_CACHE_DIR",
    str(Path(__file__).resolve().parent / "runtime_cache"),
)
DEFAULT_DEVICE = os.getenv("PFSA_APP_DEVICE", "auto")

SYNTHETIC_EXAMPLES = {
    "Indirect + discourse coreference": {
        "doc_id": "demo_indirect_coref",
        "text": (
            "Dalam rapat kebijakan di Jakarta pada Senin, Ketua Komisi Digital Raka Pratama "
            "mengatakan pemerintah perlu memperkuat perlindungan data masyarakat. "
            "Ia menambahkan bahwa regulasi baru harus memberikan mekanisme pengaduan yang jelas. "
            "Menurutnya, transparansi penggunaan kecerdasan artifisial juga perlu ditingkatkan."
        ),
    },
    "Direct + contextual arguments": {
        "doc_id": "demo_direct",
        "text": (
            "Direktur Pusat Kebijakan Digital Maya Santoso menghadiri konferensi teknologi di Yogyakarta "
            "pada Selasa. \"Kami akan membuka hasil evaluasi sistem kepada publik,\" katanya. "
            "Maya menjelaskan bahwa keterbukaan tersebut diperlukan untuk meningkatkan akuntabilitas."
        ),
    },
    "Multiple speakers": {
        "doc_id": "demo_multi_speaker",
        "text": (
            "Ketua Forum Teknologi Bima Wirawan menilai tata kelola AI perlu memiliki standar audit yang jelas. "
            "Ia meminta lembaga publik menyimpan bukti keputusan otomatis. "
            "Sementara itu, peneliti kebijakan Nita Prameswari mengatakan dokumentasi sumber data juga penting. "
            "Menurutnya, setiap keluaran sistem harus dapat ditelusuri kembali ke bukti tekstual."
        ),
    },
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px;}
    .hero {
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, rgba(15,23,42,.035), rgba(37,99,235,.045));
    }
    .hero h1 {margin: 0 0 .25rem 0; font-size: 2rem;}
    .hero p {margin: 0; color: #536171;}
    .step-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0,1fr));
        gap: .65rem;
        margin: .5rem 0 1.1rem 0;
    }
    .step-card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: .75rem .9rem;
        min-height: 74px;
        background: rgba(255,255,255,.65);
    }
    .step-n {font-size: .72rem; color: #64748b; text-transform: uppercase; letter-spacing: .04em;}
    .step-t {font-size: .95rem; font-weight: 700; margin-top: .15rem;}
    .muted {color: #64748b;}
    .evidence-box {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        line-height: 1.9;
        font-size: 1rem;
        background: #fff;
    }
    .ev-statement {background:#dbeafe; border-bottom:2px solid #2563eb; padding:.08rem .12rem; border-radius:4px;}
    .ev-cue {background:#fef3c7; border-bottom:2px solid #d97706; padding:.08rem .12rem; border-radius:4px;}
    .ev-speaker {background:#dcfce7; border-bottom:2px solid #16a34a; padding:.08rem .12rem; border-radius:4px;}
    .ev-role {background:#f3e8ff; border-bottom:2px solid #9333ea; padding:.08rem .12rem; border-radius:4px;}
    .ev-affiliation {background:#fae8ff; border-bottom:2px solid #c026d3; padding:.08rem .12rem; border-radius:4px;}
    .ev-datetime {background:#cffafe; border-bottom:2px solid #0891b2; padding:.08rem .12rem; border-radius:4px;}
    .ev-location {background:#ffedd5; border-bottom:2px solid #ea580c; padding:.08rem .12rem; border-radius:4px;}
    .ev-event {background:#f1f5f9; border-bottom:2px solid #475569; padding:.08rem .12rem; border-radius:4px;}
    .ev-issue {background:#ffe4e6; border-bottom:2px solid #e11d48; padding:.08rem .12rem; border-radius:4px;}
    .legend-chip {
        display:inline-block; margin:.12rem .2rem .12rem 0; padding:.18rem .45rem;
        border-radius:999px; border:1px solid #d1d5db; font-size:.78rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: .65rem .8rem;
    }
    @media (max-width: 900px) {.step-row {grid-template-columns: 1fr 1fr;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "bundle": None,
        "bundle_key": None,
        "single_result": None,
        "batch_result": None,
        "news_text": SYNTHETIC_EXAMPLES["Indirect + discourse coreference"]["text"],
        "doc_id_input": SYNTHETIC_EXAMPLES["Indirect + discourse coreference"]["doc_id"],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def _clear_loaded_bundle() -> None:
    bundle = st.session_state.get("bundle")
    if bundle is not None:
        release_model_bundle(bundle)
    st.session_state.bundle = None
    st.session_state.bundle_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".csv"):
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(
                    io.BytesIO(data),
                    sep=None,
                    engine="python",
                    encoding=encoding,
                )
            except Exception:
                continue
        raise ValueError("CSV could not be decoded.")

    if name.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(data), engine="openpyxl")

    if name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))

    raise ValueError("Supported formats: CSV, XLSX, XLSM, XLS.")


def _device_label(row: dict) -> str:
    if row["device"] == "cpu":
        return "CPU"
    free = "?" if row.get("free_gb") is None else f"{row['free_gb']:.2f} GiB free"
    total = "?" if row.get("total_gb") is None else f"{row['total_gb']:.2f} GiB"
    return f"{row['device']} · {row['name']} · {free} / {total}"


def _node_label(data: dict, node_id: str) -> str:
    label = str(data.get("label") or data.get("content") or node_id)
    if data.get("node_type") == "StatementEvent":
        statement_type = data.get("statement_type", "")
        prefix = "D" if statement_type == "DIRECT" else "I"
        label = f"[{prefix}] {label}"
    return label if len(label) <= 78 else label[:75] + "..."


def _graph_subgraph(
    graph: nx.MultiDiGraph,
    focus_event_node: Optional[str],
    selected_relations: List[str],
    max_nodes: int,
) -> nx.MultiDiGraph:
    filtered = nx.MultiDiGraph()

    selected_set = set(selected_relations)
    for node, attrs in graph.nodes(data=True):
        filtered.add_node(node, **attrs)
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        relation = attrs.get("relation")
        if not selected_set or relation in selected_set:
            filtered.add_edge(source, target, key=key, **attrs)

    isolates = list(nx.isolates(filtered))
    filtered.remove_nodes_from(isolates)

    if focus_event_node and focus_event_node in filtered:
        keep = {focus_event_node}
        keep.update(filtered.predecessors(focus_event_node))
        keep.update(filtered.successors(focus_event_node))
        return filtered.subgraph(keep).copy()

    if filtered.number_of_nodes() <= max_nodes:
        return filtered

    event_nodes = [
        node
        for node, attrs in filtered.nodes(data=True)
        if attrs.get("node_type") == "StatementEvent"
    ]
    keep = set(event_nodes[: max(1, max_nodes // 4)])
    for event_node in list(keep):
        keep.update(filtered.predecessors(event_node))
        keep.update(filtered.successors(event_node))
        if len(keep) >= max_nodes:
            break

    ordered = [node for node in filtered.nodes if node in keep][:max_nodes]
    return filtered.subgraph(ordered).copy()


def _graph_to_pyvis_html(
    graph: nx.MultiDiGraph,
    *,
    height: int = 680,
    physics: bool = True,
) -> str:
    if graph.number_of_nodes() == 0:
        return "<div style='padding:1rem'>No graph nodes to display.</div>"

    net = Network(
        height=f"{height}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1f2937",
        cdn_resources="in_line",
    )

    shape_map = {
        "Article": "box",
        "StatementEvent": "dot",
        "Person": "ellipse",
        "Role": "ellipse",
        "Organization": "ellipse",
        "DateTime": "ellipse",
        "Location": "ellipse",
        "UtteranceEvent": "ellipse",
        "Issue": "ellipse",
    }
    color_map = {
        "Article": "#e2e8f0",
        "StatementEvent": "#bfdbfe",
        "Person": "#bbf7d0",
        "Role": "#e9d5ff",
        "Organization": "#f5d0fe",
        "DateTime": "#a5f3fc",
        "Location": "#fed7aa",
        "UtteranceEvent": "#cbd5e1",
        "Issue": "#fecdd3",
    }

    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "Entity")
        title_parts = [
            f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}"
            for k, v in attrs.items()
            if v not in (None, "")
        ]
        size = 28 if node_type == "StatementEvent" else 20
        net.add_node(
            node_id,
            label=_node_label(attrs, node_id),
            title="<br>".join(title_parts),
            shape=shape_map.get(node_type, "ellipse"),
            color=color_map.get(node_type, "#e5e7eb"),
            size=size,
            borderWidth=2 if node_type == "StatementEvent" else 1,
        )

    for source, target, _key, attrs in graph.edges(keys=True, data=True):
        relation = str(attrs.get("relation", ""))
        relation_confidence = attrs.get("relation_confidence")
        label = relation
        if relation_confidence is not None:
            label = f"{relation} ({float(relation_confidence):.2f})"

        title_parts = [
            f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}"
            for k, v in attrs.items()
            if v not in (None, "")
        ]
        net.add_edge(
            source,
            target,
            label=label,
            title="<br>".join(title_parts),
            arrows="to",
        )

    if physics:
        net.barnes_hut(
            gravity=-4200,
            central_gravity=0.18,
            spring_length=160,
            spring_strength=0.045,
            damping=0.9,
        )
    else:
        net.toggle_physics(False)

    return net.generate_html(notebook=False)


def _event_options(events: List[dict]) -> Dict[str, dict]:
    options = {}
    for idx, event in enumerate(events, start=1):
        statement = (event.get("statement") or {}).get("text", "")
        speaker = (
            (event.get("speaker") or {}).get("canonical")
            or (event.get("speaker") or {}).get("text")
            or "speaker?"
        )
        short = statement if len(statement) <= 90 else statement[:87] + "..."
        key = f"{idx}. {event.get('statement_type')} · {speaker} · {short}"
        options[key] = event
    return options


def _event_spans(event: dict) -> List[dict]:
    mapping = [
        ("statement", "STATEMENT", "ev-statement"),
        ("cue", "CUE", "ev-cue"),
        ("speaker", "SPEAKER", "ev-speaker"),
        ("role", "ROLE", "ev-role"),
        ("affiliation", "AFFILIATION", "ev-affiliation"),
        ("datetime", "DATETIME", "ev-datetime"),
        ("location", "LOCATION", "ev-location"),
        ("utterance_event", "EVENT", "ev-event"),
        ("issue", "ISSUE", "ev-issue"),
    ]
    spans = []
    for field, label, css_class in mapping:
        obj = event.get(field)
        if not obj:
            continue
        start = obj.get("start")
        end = obj.get("end")
        if start is None or end is None:
            continue
        spans.append(
            {
                "field": field,
                "label": label,
                "css_class": css_class,
                "start": int(start),
                "end": int(end),
            }
        )
    return spans


def _highlight_event_text(text: str, event: dict) -> str:
    spans = sorted(
        _event_spans(event),
        key=lambda x: (x["start"], -(x["end"] - x["start"])),
    )

    # Evidence spans can theoretically overlap. For visualization, retain the
    # first span at a position and skip overlapping lower-priority spans.
    accepted = []
    cursor = -1
    for span in spans:
        if span["start"] < 0 or span["end"] > len(text) or span["start"] >= span["end"]:
            continue
        if span["start"] < cursor:
            continue
        accepted.append(span)
        cursor = span["end"]

    parts = []
    cursor = 0
    for span in accepted:
        parts.append(html.escape(text[cursor : span["start"]]))
        segment = html.escape(text[span["start"] : span["end"]])
        parts.append(
            f'<span class="{span["css_class"]}" '
            f'title="{html.escape(span["label"])}">{segment}</span>'
        )
        cursor = span["end"]
    parts.append(html.escape(text[cursor:]))
    return "".join(parts).replace("\n", "<br>")


def _render_legend() -> None:
    st.markdown(
        """
        <span class="legend-chip ev-statement">Statement</span>
        <span class="legend-chip ev-cue">Cue</span>
        <span class="legend-chip ev-speaker">Speaker</span>
        <span class="legend-chip ev-role">Role</span>
        <span class="legend-chip ev-affiliation">Affiliation</span>
        <span class="legend-chip ev-datetime">Date/time</span>
        <span class="legend-chip ev-location">Location</span>
        <span class="legend-chip ev-event">Event</span>
        <span class="legend-chip ev-issue">Issue</span>
        """,
        unsafe_allow_html=True,
    )


def _render_event_card(event: dict) -> None:
    statement = event.get("statement") or {}
    cue = event.get("cue") or {}
    speaker = event.get("speaker") or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type", event.get("statement_type", "—"))
    c2.metric("Statement confidence", f"{float(statement.get('confidence', 0.0)):.3f}")
    c3.metric("Cue label", cue.get("label") or "—")
    relation_conf = speaker.get("relation_confidence")
    c4.metric(
        "Speaker relation",
        "—" if relation_conf is None else f"{float(relation_conf):.3f}",
    )

    st.markdown("**Statement content**")
    st.write(statement.get("text") or "—")

    summary_rows = [
        ("Cue", cue.get("text"), cue.get("label")),
        (
            "Speaker",
            speaker.get("canonical") or speaker.get("text"),
            speaker.get("mention_label") or speaker.get("source_mention_label"),
        ),
        ("Role", (event.get("role") or {}).get("text"), None),
        ("Affiliation", (event.get("affiliation") or {}).get("text"), None),
        ("Date/time", (event.get("datetime") or {}).get("text"), None),
        ("Location", (event.get("location") or {}).get("text"), None),
        ("Utterance event", (event.get("utterance_event") or {}).get("text"), None),
        ("Issue", (event.get("issue") or {}).get("text"), None),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Element", "Value", "Label"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    relation_df = pd.DataFrame(relation_rows_for_event(event))
    if not relation_df.empty:
        st.markdown("**Predicted relations**")
        st.dataframe(relation_df, use_container_width=True, hide_index=True)


def _render_kg(
    events: List[dict],
    *,
    default_focus_event: Optional[dict] = None,
    key_prefix: str,
) -> None:
    graph = build_nx_kg(events)

    m1, m2, m3 = st.columns(3)
    m1.metric("KG nodes", graph.number_of_nodes())
    m2.metric("KG edges", graph.number_of_edges())
    m3.metric(
        "Statement-event nodes",
        sum(
            attrs.get("node_type") == "StatementEvent"
            for _, attrs in graph.nodes(data=True)
        ),
    )

    relation_types = sorted(
        {
            attrs.get("relation")
            for _, _, _, attrs in graph.edges(keys=True, data=True)
            if attrs.get("relation")
        }
    )

    controls1, controls2, controls3 = st.columns([2, 2, 1])
    selected_relations = controls1.multiselect(
        "Relations",
        relation_types,
        default=relation_types,
        key=f"{key_prefix}_relations",
    )

    event_options = {"All statement events": None}
    event_options.update(_event_options(events))
    default_index = 0
    if default_focus_event:
        for idx, value in enumerate(event_options.values()):
            if value and value.get("event_id") == default_focus_event.get("event_id"):
                default_index = idx
                break

    focus_label = controls2.selectbox(
        "Graph focus",
        list(event_options.keys()),
        index=default_index,
        key=f"{key_prefix}_focus",
    )
    focus_event = event_options[focus_label]
    max_nodes = controls3.number_input(
        "Max nodes",
        min_value=20,
        max_value=500,
        value=160,
        step=20,
        key=f"{key_prefix}_max_nodes",
    )
    physics = st.checkbox(
        "Interactive graph physics",
        value=True,
        key=f"{key_prefix}_physics",
    )

    focus_node = (
        None
        if focus_event is None
        else f"statement_event:{focus_event['event_id']}"
    )
    display_graph = _graph_subgraph(
        graph,
        focus_event_node=focus_node,
        selected_relations=selected_relations,
        max_nodes=int(max_nodes),
    )

    st.caption(
        "StatementEvent is the central node. Cue/CUECOREF remains evidence metadata "
        "on the StatementEvent; formal KG edges represent source/context relations."
    )
    components.html(
        _graph_to_pyvis_html(display_graph, physics=physics),
        height=710,
        scrolling=True,
    )

    with st.expander("Visible graph edges"):
        edge_df = graph_edges_dataframe(display_graph)
        st.dataframe(edge_df, use_container_width=True, hide_index=True)


def _render_downloads(events: List[dict], prefix: str) -> None:
    event_df = events_to_dataframe(events)
    graph = build_nx_kg(events)
    edge_df = graph_edges_dataframe(graph)

    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Events JSONL",
        events_to_jsonl_bytes(events),
        file_name=f"{prefix}_statement_events.jsonl",
        mime="application/x-ndjson",
        use_container_width=True,
    )
    d2.download_button(
        "Events CSV",
        dataframe_to_csv_bytes(event_df),
        file_name=f"{prefix}_statement_events.csv",
        mime="text/csv",
        use_container_width=True,
    )
    d3.download_button(
        "KG edge list CSV",
        dataframe_to_csv_bytes(edge_df),
        file_name=f"{prefix}_kg_edges.csv",
        mime="text/csv",
        use_container_width=True,
    )

    d4, d5, d6 = st.columns(3)
    d4.download_button(
        "GraphML",
        graphml_bytes(graph),
        file_name=f"{prefix}_kg.graphml",
        mime="application/xml",
        use_container_width=True,
    )
    d5.download_button(
        "RDF / Turtle",
        rdf_bytes(events, "turtle"),
        file_name=f"{prefix}_kg.ttl",
        mime="text/turtle",
        use_container_width=True,
    )
    d6.download_button(
        "JSON-LD",
        rdf_bytes(events, "json-ld"),
        file_name=f"{prefix}_kg.jsonld",
        mime="application/ld+json",
        use_container_width=True,
    )


def _render_result(
    result: dict,
    *,
    key_prefix: str,
    show_document_selector: bool = False,
) -> None:
    all_events = result["events"]
    spans_by_doc = result["spans_by_doc"]
    texts_by_doc = result["texts_by_doc"]

    if show_document_selector:
        doc_ids = sorted(texts_by_doc)
        if not doc_ids:
            selected_doc = None
            events = all_events
            spans = []
            text = ""
        else:
            graph_scope = st.radio(
                "Result scope",
                ["One document", "All documents"],
                horizontal=True,
                key=f"{key_prefix}_scope",
            )
            if graph_scope == "One document":
                selected_doc = st.selectbox(
                    "Document",
                    doc_ids,
                    key=f"{key_prefix}_doc",
                )
                events = [
                    event for event in all_events
                    if str(event.get("doc_id")) == str(selected_doc)
                ]
                spans = spans_by_doc.get(selected_doc, [])
                text = texts_by_doc.get(selected_doc, "")
            else:
                selected_doc = None
                events = all_events
                spans = []
                text = ""
    else:
        selected_doc = next(iter(texts_by_doc), None)
        events = all_events
        spans = spans_by_doc.get(selected_doc, []) if selected_doc else []
        text = texts_by_doc.get(selected_doc, "") if selected_doc else ""

    event_df = events_to_dataframe(events)
    direct_count = (
        int((event_df["statement_type"] == "DIRECT").sum())
        if not event_df.empty
        else 0
    )
    indirect_count = (
        int((event_df["statement_type"] == "INDIRECT").sum())
        if not event_df.empty
        else 0
    )
    unique_speakers = (
        int(event_df["speaker"].dropna().nunique())
        if not event_df.empty and "speaker" in event_df
        else 0
    )

    metrics = st.columns(5)
    metrics[0].metric("Statement events", len(events))
    metrics[1].metric("Direct", direct_count)
    metrics[2].metric("Indirect", indirect_count)
    metrics[3].metric("Unique speakers", unique_speakers)
    metrics[4].metric("Inference", f"{result.get('elapsed_seconds', 0.0):.2f}s")

    overview_tab, evidence_tab, kg_tab, data_tab, provenance_tab = st.tabs(
        [
            "Overview",
            "Evidence & Attribution",
            "Knowledge Graph",
            "Data & Export",
            "Model & Provenance",
        ]
    )

    with overview_tab:
        st.markdown(
            """
            <div class="step-row">
              <div class="step-card"><div class="step-n">Step 1</div><div class="step-t">News input</div><div class="muted">Full Indonesian article</div></div>
              <div class="step-card"><div class="step-n">Step 2</div><div class="step-t">Student-model inference</div><div class="muted">BIO spans + relation heads</div></div>
              <div class="step-card"><div class="step-n">Step 3</div><div class="step-t">StatementEvent assembly</div><div class="muted">Source, cue, content, context</div></div>
              <div class="step-card"><div class="step-n">Step 4</div><div class="step-t">Evidence-grounded KG</div><div class="muted">Event-centric graph + provenance</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if event_df.empty:
            st.info(
                "No statement event passed the current model predictions and relation threshold. "
                "The application does not fabricate graph entities when extraction is empty."
            )
        else:
            visible_cols = [
                "statement_type",
                "statement",
                "statement_confidence",
                "cue",
                "cue_label",
                "speaker",
                "speaker_mention_label",
                "speaker_relation_confidence",
                "role",
                "affiliation",
                "datetime",
                "location",
                "utterance_event",
                "issue",
            ]
            st.dataframe(
                event_df[[c for c in visible_cols if c in event_df.columns]],
                use_container_width=True,
                hide_index=True,
            )

        if result.get("cache_hits") is not None:
            st.caption(
                f"Disk-cache hits: {result.get('cache_hits', 0)} / "
                f"{result.get('processed_documents', 1)} documents."
            )
        elif result.get("cache_hit") is not None:
            st.caption(
                "Inference source: disk cache"
                if result["cache_hit"]
                else "Inference source: live student-model computation"
            )

    with evidence_tab:
        if show_document_selector and selected_doc is None:
            st.info("Choose 'One document' above to inspect character-level evidence.")
        elif not events:
            st.info("No statement events are available for evidence inspection.")
        else:
            options = _event_options(events)
            selected_label = st.selectbox(
                "Statement event",
                list(options.keys()),
                key=f"{key_prefix}_event",
            )
            selected_event = options[selected_label]

            _render_event_card(selected_event)
            st.markdown("**Evidence in the original article**")
            _render_legend()
            st.markdown(
                f'<div class="evidence-box">{_highlight_event_text(text, selected_event)}</div>',
                unsafe_allow_html=True,
            )

            with st.expander("Raw event JSON"):
                st.json(selected_event)

            with st.expander("All detected BIO spans"):
                span_df = spans_to_dataframe(spans, selected_doc)
                if span_df.empty:
                    st.info("No spans were detected.")
                else:
                    span_types = sorted(span_df["type"].dropna().unique())
                    selected_types = st.multiselect(
                        "Span types",
                        span_types,
                        default=span_types,
                        key=f"{key_prefix}_span_types",
                    )
                    if selected_types:
                        span_df = span_df[span_df["type"].isin(selected_types)]
                    st.dataframe(
                        span_df,
                        use_container_width=True,
                        hide_index=True,
                    )

    with kg_tab:
        _render_kg(
            events,
            key_prefix=f"{key_prefix}_kg",
        )

    with data_tab:
        st.markdown("**StatementEvent table**")
        st.dataframe(
            event_df,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Formal KG edge table**")
        graph = build_nx_kg(events)
        edge_df = graph_edges_dataframe(graph)
        st.dataframe(
            edge_df,
            use_container_width=True,
            hide_index=True,
        )
        _render_downloads(
            events,
            prefix=(
                str(selected_doc)
                if selected_doc
                else f"{key_prefix}_all_documents"
            ),
        )

    with provenance_tab:
        bundle = st.session_state.get("bundle")
        model_info = {}
        if bundle is not None:
            model_info = {
                "model_id": bundle.model_id,
                "resolved_revision": bundle.resolved_revision,
                "device": str(bundle.device),
                "checkpoint_sha256": bundle.checkpoint_sha256,
                "training_fingerprint": bundle.manifest.get("training_fingerprint"),
                "manifest_schema_version": bundle.manifest.get("manifest_schema_version"),
                "best_epoch": bundle.manifest.get("best_epoch"),
                "best_validation_loss": bundle.manifest.get("best_validation_loss"),
                "max_length": bundle.settings.max_length,
                "stride": bundle.settings.stride,
                "app_inference_schema_version": APP_INFERENCE_SCHEMA_VERSION,
            }
        st.json(model_info)

        st.info(
            "Deployment inference uses only the trained document-level student model. "
            "LLM annotation and weak supervision are training-time procedures and are not "
            "called by this Streamlit application."
        )
        st.caption(
            "Statement confidence comes from the BIO token classifier. Relation confidence "
            "comes from the trained relation head. PERSONCOREF canonicalization is a conservative "
            "document-level deployment heuristic and its provenance is retained in the event/KG metadata."
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="hero">
      <h1>{APP_TITLE}</h1>
      <p>{APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="step-row">
      <div class="step-card"><div class="step-n">1 · Input</div><div class="step-t">News article</div></div>
      <div class="step-card"><div class="step-n">2 · AI</div><div class="step-t">Document inference</div></div>
      <div class="step-card"><div class="step-n">3 · IE</div><div class="step-t">StatementEvent extraction</div></div>
      <div class="step-card"><div class="step-n">4 · KG</div><div class="step-t">Interactive graph</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar model control
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Deployment model")

    artifact_dir = st.text_input(
        "V14 model artifact directory",
        value=DEFAULT_ARTIFACT_DIR,
        help=(
            "Directory containing student_model_best.pt, student_model_manifest.json, "
            "student_tokenizer/, and preferably student_encoder_config/."
        ),
    )

    status = inspect_artifact_dir(Path(artifact_dir))
    if status["ready"]:
        st.success("V14-compatible model artifact verified.")
    else:
        st.warning("Model artifact is not ready.")
        for problem in status.get("problems", []):
            st.caption(f"• {problem}")

    with st.expander("Artifact audit"):
        st.json(status)

    devices = list_runtime_devices()
    device_values = ["auto"] + [row["device"] for row in devices]
    if DEFAULT_DEVICE not in device_values:
        default_device_index = 0
    else:
        default_device_index = device_values.index(DEFAULT_DEVICE)

    chosen_device = st.selectbox(
        "Inference device",
        device_values,
        index=default_device_index,
        format_func=lambda value: (
            "Auto · CUDA if available, otherwise CPU"
            if value == "auto"
            else _device_label(next(row for row in devices if row["device"] == value))
        ),
    )

    if torch.cuda.is_available():
        with st.expander("Visible GPU memory"):
            st.dataframe(
                pd.DataFrame([row for row in devices if row["device"] != "cpu"]),
                use_container_width=True,
                hide_index=True,
            )

    relation_threshold = st.slider(
        "Relation confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Only argument relations at or above this probability are attached to StatementEvent.",
    )

    use_inference_cache = st.checkbox(
        "Reuse inference cache",
        value=True,
    )
    cache_dir = st.text_input(
        "Inference cache directory",
        value=DEFAULT_CACHE_DIR,
    )

    desired_bundle_key = (str(Path(artifact_dir).resolve()), chosen_device)
    loaded_key = st.session_state.get("bundle_key")
    loaded = st.session_state.get("bundle") is not None

    if loaded and loaded_key == desired_bundle_key:
        st.success("Model loaded.")
        st.caption(f"Device: `{st.session_state.bundle.device}`")
    elif loaded:
        st.warning("Model settings changed. Reload before inference.")

    load_col, release_col = st.columns(2)
    if load_col.button(
        "Load / Reload",
        type="primary",
        disabled=not status["ready"],
        use_container_width=True,
    ):
        _clear_loaded_bundle()
        try:
            with st.spinner("Loading verified V14 student model..."):
                bundle = load_model_bundle(
                    Path(artifact_dir),
                    relation_threshold=relation_threshold,
                    device=chosen_device,
                )
            st.session_state.bundle = bundle
            st.session_state.bundle_key = desired_bundle_key
            st.success("Model loaded successfully.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load model: {exc}")

    if release_col.button(
        "Release",
        disabled=not loaded,
        use_container_width=True,
    ):
        _clear_loaded_bundle()
        st.rerun()

    st.divider()
    st.caption(
        "The application never retrains the model and never calls the LLM. "
        "It loads a verified trained checkpoint and performs student-model inference only."
    )


bundle_ready = (
    st.session_state.get("bundle") is not None
    and st.session_state.get("bundle_key") == desired_bundle_key
)


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

single_tab, batch_tab, method_tab = st.tabs(
    ["Live Demo · One Article", "Batch Demo", "Method & KG Semantics"]
)

with single_tab:
    input_col, run_col = st.columns([3, 1])

    with input_col:
        example_name = st.selectbox(
            "Synthetic demonstration article",
            list(SYNTHETIC_EXAMPLES.keys()),
        )
    with run_col:
        st.write("")
        st.write("")
        if st.button("Load example", use_container_width=True):
            st.session_state.news_text = SYNTHETIC_EXAMPLES[example_name]["text"]
            st.session_state.doc_id_input = SYNTHETIC_EXAMPLES[example_name]["doc_id"]
            st.rerun()

    st.caption(
        "Built-in examples are synthetic demonstration text, not factual news reports. "
        "You can replace them with any Indonesian news article."
    )

    st.text_input(
        "Document ID",
        key="doc_id_input",
        help="Optional. If blank, a stable ID is generated from the article text.",
    )
    st.text_area(
        "News article",
        key="news_text",
        height=260,
        placeholder="Paste a complete Indonesian news article here...",
    )

    run_disabled = not bundle_ready
    if not bundle_ready:
        st.info("Load a verified V14 model from the sidebar to enable live inference.")

    if st.button(
        "Run inference → build knowledge graph",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    ):
        text = st.session_state.news_text
        if not str(text).strip():
            st.warning("Enter news text first.")
        else:
            doc_id = normalize_doc_id(
                st.session_state.doc_id_input,
                text,
            )
            started = time.perf_counter()
            try:
                with st.spinner(
                    "Running document-level span extraction, attribution, and KG construction..."
                ):
                    events, spans, cache_hit = infer_with_disk_cache(
                        text,
                        doc_id,
                        st.session_state.bundle,
                        relation_threshold,
                        Path(cache_dir),
                        use_cache=use_inference_cache,
                    )
                elapsed = time.perf_counter() - started
                st.session_state.single_result = {
                    "events": events,
                    "spans_by_doc": {doc_id: spans},
                    "texts_by_doc": {doc_id: text},
                    "elapsed_seconds": elapsed,
                    "cache_hit": cache_hit,
                    "processed_documents": 1,
                }
            except torch.cuda.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.error(
                    "CUDA ran out of memory during inference. Choose another visible GPU "
                    "or CPU in the sidebar, then reload the model."
                )
                with st.expander("CUDA error detail"):
                    st.exception(exc)
            except Exception as exc:
                st.exception(exc)

    if st.session_state.single_result is not None:
        st.divider()
        _render_result(
            st.session_state.single_result,
            key_prefix="single",
            show_document_selector=False,
        )

with batch_tab:
    st.write(
        "Upload multiple news articles, run the same verified student model on each document, "
        "and inspect either a per-document graph or an aggregate model-generated KG."
    )

    uploaded = st.file_uploader(
        "CSV / Excel",
        type=["csv", "xlsx", "xlsm", "xls"],
    )

    if uploaded is not None:
        try:
            batch_df = _read_uploaded_table(uploaded)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            batch_df = None

        if batch_df is not None:
            st.dataframe(
                batch_df.head(10),
                use_container_width=True,
                hide_index=True,
            )
            columns = list(batch_df.columns)
            text_column = st.selectbox(
                "Article text column",
                columns,
            )
            id_options = ["<auto>"] + columns
            id_column = st.selectbox(
                "Document ID column",
                id_options,
            )

            if not bundle_ready:
                st.info("Load the V14 model from the sidebar to enable batch inference.")

            if st.button(
                "Run batch inference",
                type="primary",
                disabled=not bundle_ready,
            ):
                started = time.perf_counter()
                all_events: List[dict] = []
                spans_by_doc: Dict[str, List[dict]] = {}
                texts_by_doc: Dict[str, str] = {}
                summaries = []
                cache_hits = 0

                progress = st.progress(0.0, text="Starting batch inference...")
                status_box = st.empty()

                for position, (row_index, row) in enumerate(
                    batch_df.iterrows(),
                    start=1,
                ):
                    raw_text = row.get(text_column)
                    text = "" if pd.isna(raw_text) else str(raw_text)
                    if not text.strip():
                        summaries.append(
                            {
                                "row": row_index,
                                "doc_id": None,
                                "status": "EMPTY_TEXT",
                                "statement_events": 0,
                            }
                        )
                        progress.progress(
                            position / max(1, len(batch_df)),
                            text=f"Processed {position}/{len(batch_df)}",
                        )
                        continue

                    raw_id = None if id_column == "<auto>" else row.get(id_column)
                    doc_id = normalize_doc_id(
                        raw_id,
                        text,
                        index=position,
                    )
                    status_box.caption(
                        f"Inference {position}/{len(batch_df)} · {doc_id}"
                    )

                    try:
                        events, spans, cache_hit = infer_with_disk_cache(
                            text,
                            doc_id,
                            st.session_state.bundle,
                            relation_threshold,
                            Path(cache_dir),
                            use_cache=use_inference_cache,
                        )
                        all_events.extend(events)
                        spans_by_doc[doc_id] = spans
                        texts_by_doc[doc_id] = text
                        cache_hits += int(cache_hit)
                        summaries.append(
                            {
                                "row": row_index,
                                "doc_id": doc_id,
                                "status": "OK_CACHE" if cache_hit else "OK_MODEL",
                                "statement_events": len(events),
                                "direct": sum(
                                    event.get("statement_type") == "DIRECT"
                                    for event in events
                                ),
                                "indirect": sum(
                                    event.get("statement_type") == "INDIRECT"
                                    for event in events
                                ),
                            }
                        )
                    except Exception as exc:
                        summaries.append(
                            {
                                "row": row_index,
                                "doc_id": doc_id,
                                "status": f"ERROR: {exc}",
                                "statement_events": 0,
                            }
                        )

                    progress.progress(
                        position / max(1, len(batch_df)),
                        text=f"Processed {position}/{len(batch_df)}",
                    )

                status_box.empty()
                elapsed = time.perf_counter() - started
                st.session_state.batch_result = {
                    "events": all_events,
                    "spans_by_doc": spans_by_doc,
                    "texts_by_doc": texts_by_doc,
                    "elapsed_seconds": elapsed,
                    "cache_hits": cache_hits,
                    "processed_documents": len(batch_df),
                    "summary": summaries,
                }

    if st.session_state.batch_result is not None:
        result = st.session_state.batch_result
        st.success(
            f"Batch completed · {result.get('processed_documents', 0)} rows · "
            f"{result.get('cache_hits', 0)} cache hits."
        )
        summary_df = pd.DataFrame(result.get("summary", []))
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download batch processing summary",
            dataframe_to_csv_bytes(summary_df),
            file_name="pfsa_batch_inference_summary.csv",
            mime="text/csv",
        )
        st.divider()
        _render_result(
            result,
            key_prefix="batch",
            show_document_selector=True,
        )

with method_tab:
    st.subheader("What the demo actually executes")
    st.markdown(
        """
        **Deployment path**

        `News article → trained document-level student model → span predictions → relation predictions → StatementEvent objects → evidence-grounded KG`

        The application does **not** execute the training-time LLM annotator or weak-supervision label model.
        This separation is intentional: the Streamlit demo represents the deployable inference path of the final
        student model.
        """
    )

    st.subheader("V14 extraction schema")
    schema_df = pd.DataFrame(
        [
            ("Span", "STATEMENT_DIRECT / STATEMENT_INDIRECT", "Statement content"),
            ("Source", "PERSON / PERSONCOREF", "Attributed public-figure source"),
            ("Cue", "CUE / CUECOREF", "Reporting / speech-act evidence"),
            ("Context", "ROLE", "Source role"),
            ("Context", "AFFILIATION", "Source organization"),
            ("Context", "DATETIME", "Statement time"),
            ("Context", "LOCATION", "Statement location"),
            ("Context", "EVENT", "Utterance event"),
            ("Context", "ISSUE", "Issue/topic evidence"),
        ],
        columns=["Layer", "Label", "Meaning"],
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    st.subheader("Formal knowledge-graph relations")
    relation_df = pd.DataFrame(
        [
            ("StatementEvent", "ATTRIBUTED_TO", "Person"),
            ("StatementEvent", "HAS_ROLE", "Role"),
            ("StatementEvent", "AFFILIATED_WITH", "Organization"),
            ("StatementEvent", "AT_TIME", "DateTime"),
            ("StatementEvent", "AT_LOCATION", "Location"),
            ("StatementEvent", "AT_EVENT", "UtteranceEvent"),
            ("StatementEvent", "ABOUT_ISSUE", "Issue"),
            ("StatementEvent", "SOURCE_ARTICLE", "Article"),
        ],
        columns=["Source node", "Relation", "Target node"],
    )
    st.dataframe(relation_df, use_container_width=True, hide_index=True)

    st.caption(
        "CUE/CUECOREF is visualized as evidence metadata attached to StatementEvent rather "
        "than being promoted to a separate formal KG relation. PERSONCOREF remains distinct "
        "at the extraction layer, while the graph uses its resolved canonical person when available."
    )
