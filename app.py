from __future__ import annotations

import gc
import hashlib
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

EXAMPLES = {
    "Direct + indirect statements": {
        "doc_id": "demo_direct_indirect",
        "text": (
            "Kepala Pusat Kebijakan Digital Arif Nugraha mengatakan, \"Audit sistem AI harus dapat ditelusuri sampai ke sumber datanya.\" "
            "Ia menegaskan bahwa lembaga publik juga perlu menyimpan bukti keputusan otomatis. "
            "Menurut Arif, dokumentasi yang lengkap penting agar masyarakat dapat memahami dasar sebuah keputusan."
        ),
    },
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
    .statement-box {border-radius: 10px; padding: .9rem 1rem; font-size: 1.05rem; line-height: 1.65; border-left: 4px solid #64748b; background: #f8fafc;}
    .statement-direct {border-left-color:#2563eb; background:#eff6ff;}
    .statement-indirect {border-left-color:#d97706; background:#fffbeb;}
    .type-badge {display:inline-block; margin:0 .35rem .45rem 0; padding:.18rem .5rem; border-radius:999px; font-size:.75rem; font-weight:700;}
    .type-direct {background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd;}
    .type-indirect {background:#fef3c7; color:#b45309; border:1px solid #fcd34d;}
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
    defaults = {
        "bundle": None,
        "bundle_key": None,
        "result": None,
        "news_text": EXAMPLES["Direct + indirect statements"]["text"],
        "doc_id": EXAMPLES["Direct + indirect statements"]["doc_id"],
        "example_name": "Direct + indirect statements",
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
steps[0].markdown('<div class="step"><div class="step-num">Step 1</div><div class="step-title">Enter news article</div><div class="step-text">Paste an Indonesian article or load an example.</div></div>', unsafe_allow_html=True)
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
                render_legend()
                st.markdown(
                    f'<div class="evidence-box">{highlighted_text(result["text"], selected_event)}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Detected BILUO spans"):
                    span_df = spans_to_dataframe(result["spans"], result["doc_id"])
                    st.dataframe(span_df, hide_index=True, use_container_width=True)
                with st.expander("Raw StatementEvent JSON"):
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
