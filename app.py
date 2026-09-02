from __future__ import annotations

import gc
import html
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
DEFAULT_ARTIFACT_DIR = os.getenv("PFSA_MODEL_ARTIFACT_DIR", str(Path(__file__).resolve().parent / "model_artifacts"))
DEFAULT_CACHE_DIR = os.getenv("PFSA_APP_CACHE_DIR", str(Path(__file__).resolve().parent / "runtime_cache"))
DEFAULT_DEVICE = os.getenv("PFSA_APP_DEVICE", "auto")

EXAMPLES = {
    "Indirect statement + source profile": {
        "doc_id": "demo_indirect_profile",
        "text": (
            "Nahdlatul Ulama di bawah kepemimpinan Rais Aam Ahmad Syakir dan Ketua Umum Hasan Mahmud "
            "diharapkan tetap menjaga jarak dengan penguasa. Pengajar Departemen Politik dan Pemerintahan "
            "Universitas Nusantara, Abdul Karim, mengatakan Nahdlatul Ulama perlu menguatkan kembali perannya "
            "sebagai elemen masyarakat sipil di tengah pelemahan demokrasi yang terjadi belakangan ini. "
            "Menurutnya, organisasi masyarakat sipil perlu menjaga daya kritis terhadap kekuasaan."
        ),
    },
    "Indirect statement + CUECOREF": {
        "doc_id": "demo_coref",
        "text": (
            "Direktur Pusat Kebijakan Digital Raka Pratama mengatakan pemerintah perlu memperkuat perlindungan "
            "data masyarakat. Ia menambahkan bahwa regulasi baru harus memberikan mekanisme pengaduan yang jelas. "
            "Menurutnya, transparansi penggunaan kecerdasan artifisial juga perlu ditingkatkan."
        ),
    },
    "Multiple speakers": {
        "doc_id": "demo_multiple_speakers",
        "text": (
            "Ketua Forum Teknologi Bima Wirawan menilai tata kelola AI perlu memiliki standar audit yang jelas. "
            "Ia meminta lembaga publik menyimpan bukti keputusan otomatis. Peneliti kebijakan Nita Prameswari "
            "mengatakan dokumentasi sumber data juga penting. Menurutnya, setiap keluaran sistem harus dapat "
            "ditelusuri kembali ke bukti tekstual."
        ),
    },
}

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
    .statement-box {border-left: 4px solid #2563eb; background: #eff6ff; border-radius: 10px; padding: .9rem 1rem; font-size: 1.05rem; line-height: 1.65;}
    .evidence-box {border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.1rem; line-height: 1.9; background: #fff;}
    .ev-statement {background:#dbeafe; border-bottom:2px solid #2563eb; padding:.06rem .1rem; border-radius:3px;}
    .ev-cue {background:#fef3c7; border-bottom:2px solid #d97706; padding:.06rem .1rem; border-radius:3px;}
    .ev-speaker {background:#dcfce7; border-bottom:2px solid #16a34a; padding:.06rem .1rem; border-radius:3px;}
    .ev-role {background:#f3e8ff; border-bottom:2px solid #9333ea; padding:.06rem .1rem; border-radius:3px;}
    .ev-affiliation {background:#fae8ff; border-bottom:2px solid #c026d3; padding:.06rem .1rem; border-radius:3px;}
    .ev-datetime {background:#cffafe; border-bottom:2px solid #0891b2; padding:.06rem .1rem; border-radius:3px;}
    .ev-location {background:#ffedd5; border-bottom:2px solid #ea580c; padding:.06rem .1rem; border-radius:3px;}
    .ev-event {background:#f1f5f9; border-bottom:2px solid #475569; padding:.06rem .1rem; border-radius:3px;}
    .ev-issue {background:#ffe4e6; border-bottom:2px solid #e11d48; padding:.06rem .1rem; border-radius:3px;}
    .legend {display:inline-block; margin:.1rem .18rem .4rem 0; padding:.16rem .42rem; border-radius:999px; border:1px solid #d1d5db; font-size:.75rem;}
    div[data-testid="stMetric"] {border: 1px solid #e2e8f0; border-radius: 12px; padding: .55rem .7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "bundle": None,
        "bundle_key": None,
        "result": None,
        "news_text": EXAMPLES["Indirect statement + source profile"]["text"],
        "doc_id": EXAMPLES["Indirect statement + source profile"]["doc_id"],
        "example_name": "Indirect statement + source profile",
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
        out[f"{i}. {event.get('statement_type', '—')} · {speaker} · {short}"] = event
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
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff", font_color="#0f172a", cdn_resources="in_line")
    shape = {"Article": "box", "StatementEvent": "dot"}
    colors = {
        "Article": "#e2e8f0", "StatementEvent": "#bfdbfe", "Person": "#bbf7d0", "Role": "#e9d5ff",
        "Organization": "#f5d0fe", "DateTime": "#a5f3fc", "Location": "#fed7aa", "UtteranceEvent": "#cbd5e1", "Issue": "#fecdd3",
    }
    for node_id, attrs in graph.nodes(data=True):
        typ = attrs.get("node_type", "Entity")
        label = str(attrs.get("label") or attrs.get("content") or node_id)
        if typ == "StatementEvent" and len(label) > 68:
            label = label[:65] + "..."
        title = "<br>".join(f"<b>{html.escape(str(k))}</b>: {html.escape(str(v))}" for k, v in attrs.items() if v not in (None, ""))
        net.add_node(node_id, label=label, title=title, shape=shape.get(typ, "ellipse"), color=colors.get(typ, "#e5e7eb"), size=28 if typ == "StatementEvent" else 20)
    for source, target, _key, attrs in graph.edges(keys=True, data=True):
        rel = attrs.get("relation", "")
        conf = attrs.get("relation_confidence")
        label = rel if conf is None else f"{rel} ({float(conf):.2f})"
        net.add_edge(source, target, label=label, arrows="to")
    net.barnes_hut(gravity=-3800, central_gravity=.18, spring_length=150, spring_strength=.045, damping=.9)
    return net.generate_html(notebook=False)


def event_detail(event):
    statement = event.get("statement") or {}
    st.markdown(f'<div class="statement-box">{html.escape(statement.get("text") or "—")}</div>', unsafe_allow_html=True)
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


st.markdown(f'<div class="hero"><h1>{APP_TITLE}</h1><p>From Indonesian news text to evidence-grounded StatementEvents and an interactive knowledge graph.</p></div>', unsafe_allow_html=True)

steps = st.columns(3)
steps[0].markdown('<div class="step"><div class="step-num">Step 1</div><div class="step-title">Enter news article</div><div class="step-text">Paste an Indonesian article or load an example.</div></div>', unsafe_allow_html=True)
steps[1].markdown('<div class="step"><div class="step-num">Step 2</div><div class="step-title">Run inference</div><div class="step-text">BILUO-CRF spans, ISSUE sentence head, and relations.</div></div>', unsafe_allow_html=True)
steps[2].markdown('<div class="step"><div class="step-num">Step 3</div><div class="step-title">Explore the KG</div><div class="step-text">Inspect evidence and model-generated graph relations.</div></div>', unsafe_allow_html=True)

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
            st.caption(f"Inference schema: {APP_INFERENCE_SCHEMA_VERSION}")
            st.caption(f"Checkpoint: {b.checkpoint_sha256[:12]}…")

bundle_ready = st.session_state.get("bundle") is not None and st.session_state.get("bundle_key") == model_key

st.subheader("1. News article")
example_col, id_col = st.columns([2, 1])
example_name = example_col.selectbox("Example", list(EXAMPLES) + ["Custom article"], index=(list(EXAMPLES) + ["Custom article"]).index(st.session_state.example_name) if st.session_state.example_name in (list(EXAMPLES) + ["Custom article"]) else 0)
if example_name != st.session_state.example_name:
    st.session_state.example_name = example_name
    if example_name in EXAMPLES:
        st.session_state.news_text = EXAMPLES[example_name]["text"]
        st.session_state.doc_id = EXAMPLES[example_name]["doc_id"]
    st.session_state.result = None
    st.rerun()

doc_id = id_col.text_input("Document ID", value=st.session_state.doc_id)
text = st.text_area("Article text", value=st.session_state.news_text, height=250, placeholder="Paste Indonesian news article here...")
st.session_state.news_text = text
st.session_state.doc_id = doc_id

analyze = st.button("Analyze article", type="primary", use_container_width=True, disabled=not bundle_ready or len(text.strip()) < 20)
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
            "events": events, "spans": spans, "text": text, "doc_id": final_doc_id,
            "elapsed": time.perf_counter() - started, "cache_hit": cache_hit,
        }
    except Exception as exc:
        st.error(f"Inference failed: {exc}")

result = st.session_state.get("result")
if result is not None:
    st.divider()
    st.subheader("2. Extraction result")
    events = result["events"]
    df = events_to_dataframe(events)
    direct = int((df["statement_type"] == "DIRECT").sum()) if not df.empty else 0
    indirect = int((df["statement_type"] == "INDIRECT").sum()) if not df.empty else 0
    speakers = int(df["speaker"].dropna().nunique()) if not df.empty else 0

    m = st.columns(5)
    m[0].metric("Statements", len(events))
    m[1].metric("Direct", direct)
    m[2].metric("Indirect", indirect)
    m[3].metric("Speakers", speakers)
    m[4].metric("Inference", f"{result['elapsed']:.2f}s")

    if not events:
        st.info("No StatementEvent was produced for this article at the current relation threshold.")
    else:
        options = event_options(events)
        selected_label = st.selectbox("Statement to inspect", list(options.keys()))
        selected_event = options[selected_label]

        extraction_tab, evidence_tab, graph_tab = st.tabs(["Extraction", "Evidence", "Knowledge Graph"])

        with extraction_tab:
            event_detail(selected_event)
            with st.expander("All extracted statements"):
                cols = ["statement_type", "statement", "statement_confidence", "cue", "cue_label", "speaker", "speaker_mention_label", "role", "affiliation", "datetime", "location", "utterance_event", "issue"]
                st.dataframe(df[[c for c in cols if c in df.columns]], hide_index=True, use_container_width=True)

        with evidence_tab:
            render_legend()
            st.markdown(f'<div class="evidence-box">{highlighted_text(result["text"], selected_event)}</div>', unsafe_allow_html=True)
            with st.expander("Detected BILUO spans"):
                st.dataframe(spans_to_dataframe(result["spans"], result["doc_id"]), hide_index=True, use_container_width=True)
            with st.expander("Raw StatementEvent JSON"):
                st.json(selected_event)

        with graph_tab:
            scope = st.radio("Graph", ["Selected statement", "All statements"], horizontal=True)
            graph_events = [selected_event] if scope == "Selected statement" else events
            graph = build_nx_kg(graph_events)
            gm = st.columns(3)
            gm[0].metric("Nodes", graph.number_of_nodes())
            gm[1].metric("Edges", graph.number_of_edges())
            gm[2].metric("StatementEvents", len(graph_events))
            components.html(graph_html(graph), height=630, scrolling=True)
            with st.expander("Graph edges"):
                st.dataframe(graph_edges_dataframe(graph), hide_index=True, use_container_width=True)

        with st.expander("Download results"):
            graph = build_nx_kg(events)
            d = st.columns(4)
            d[0].download_button("Events JSONL", events_to_jsonl_bytes(events), f"{result['doc_id']}_events.jsonl", "application/x-ndjson", use_container_width=True)
            d[1].download_button("Events CSV", dataframe_to_csv_bytes(df), f"{result['doc_id']}_events.csv", "text/csv", use_container_width=True)
            d[2].download_button("GraphML", graphml_bytes(graph), f"{result['doc_id']}_kg.graphml", "application/xml", use_container_width=True)
            d[3].download_button("RDF/Turtle", rdf_bytes(events, "turtle"), f"{result['doc_id']}_kg.ttl", "text/turtle", use_container_width=True)
            st.caption("This graph is generated only from student-model predictions; the Streamlit app does not call the training-time LLM or weak-supervision pipeline.")
