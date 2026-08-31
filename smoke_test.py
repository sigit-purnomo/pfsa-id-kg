from __future__ import annotations

from pfsa_inference import (
    REL_LABELS,
    TOKEN_ENTITY_TYPES,
    build_nx_kg,
    events_to_dataframe,
    graph_edges_dataframe,
    rdf_bytes,
)


def main():
    assert "PERSONCOREF" in TOKEN_ENTITY_TYPES
    assert "CUECOREF" in TOKEN_ENTITY_TYPES
    assert "ISSUE" in TOKEN_ENTITY_TYPES
    assert "ABOUT_ISSUE" in REL_LABELS

    event = {
        "event_id": "stmt_smoke",
        "doc_id": "demo",
        "statement_type": "INDIRECT",
        "statement": {
            "text": "transparansi perlu ditingkatkan",
            "evidence": "transparansi perlu ditingkatkan",
            "start": 80,
            "end": 111,
            "confidence": 0.93,
        },
        "cue": {
            "text": "Menurutnya",
            "evidence": "Menurutnya",
            "start": 68,
            "end": 78,
            "confidence": 0.91,
            "label": "CUECOREF",
        },
        "speaker": {
            "text": "Ia",
            "evidence": "Ia",
            "start": 40,
            "end": 42,
            "label": "PERSONCOREF",
            "canonical": "Raka Pratama",
            "mention_label": "PERSONCOREF",
            "source_mention_label": "CUECOREF",
            "coreference_via": "CUECOREF",
            "coref_resolution_provenance": "DOCUMENT_HEURISTIC_FROM_PREDICTED_PERSON",
            "relation_confidence": 0.87,
        },
        "issue": {
            "text": "transparansi AI",
            "evidence": "transparansi AI",
            "start": 112,
            "end": 127,
            "label": "ISSUE",
            "relation_confidence": 0.79,
        },
        "span_provenance": "STUDENT_MODEL_PREDICTION",
        "relation_provenance": "STUDENT_MODEL_PREDICTION",
    }

    graph = build_nx_kg([event])
    relations = {
        attrs.get("relation")
        for _, _, _, attrs in graph.edges(keys=True, data=True)
    }
    assert "ATTRIBUTED_TO" in relations
    assert "ABOUT_ISSUE" in relations
    assert graph.number_of_nodes() >= 4

    event_df = events_to_dataframe([event])
    assert event_df.iloc[0]["cue_label"] == "CUECOREF"
    assert event_df.iloc[0]["speaker_mention_label"] == "PERSONCOREF"

    edge_df = graph_edges_dataframe(graph)
    speaker_edge = edge_df[edge_df["relation"] == "ATTRIBUTED_TO"].iloc[0]
    assert speaker_edge["coreference_via"] == "CUECOREF"

    assert len(rdf_bytes([event], "turtle")) > 0
    print("PFSA-ID V14 Streamlit deployment smoke test: PASS")


if __name__ == "__main__":
    main()
