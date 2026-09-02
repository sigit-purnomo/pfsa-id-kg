from pfsa_inference import (
    STATEMENT_ENTITY_TYPES,
    build_nx_kg,
    events_to_dataframe,
    filter_events_by_statement_type,
    statement_type_counts,
)


def sample_event(event_id, statement_type, text, speaker):
    return {
        "event_id": event_id,
        "doc_id": "smoke_doc",
        "statement_type": statement_type,
        "statement_model_label": f"STATEMENT_{statement_type}",
        "statement": {
            "text": text,
            "start": 0,
            "end": len(text),
            "confidence": 0.95,
        },
        "speaker": {
            "text": speaker,
            "canonical": speaker,
            "start": 0,
            "end": len(speaker),
            "label": "PERSON",
            "mention_label": "PERSON",
            "relation_confidence": 0.93,
        },
        "cue": None,
        "role": None,
        "affiliation": None,
        "datetime": None,
        "location": None,
        "utterance_event": None,
        "issue": None,
    }


def main():
    assert STATEMENT_ENTITY_TYPES == ["STATEMENT_DIRECT", "STATEMENT_INDIRECT"]

    events = [
        sample_event("stmt_direct", "DIRECT", "Audit sistem AI harus transparan.", "Arif Nugraha"),
        sample_event("stmt_indirect", "INDIRECT", "Lembaga publik perlu menyimpan bukti keputusan otomatis.", "Arif Nugraha"),
    ]

    counts = statement_type_counts(events)
    assert counts == {"DIRECT": 1, "INDIRECT": 1, "ALL": 2}
    assert len(filter_events_by_statement_type(events, "DIRECT")) == 1
    assert len(filter_events_by_statement_type(events, "INDIRECT")) == 1
    assert len(filter_events_by_statement_type(events, "ALL")) == 2

    df = events_to_dataframe(events)
    assert set(df["statement_type"]) == {"DIRECT", "INDIRECT"}

    graph = build_nx_kg(events)
    statement_nodes = [attrs for _node, attrs in graph.nodes(data=True) if attrs.get("node_type") == "StatementEvent"]
    assert {x.get("statement_type") for x in statement_nodes} == {"DIRECT", "INDIRECT"}

    print("PASS: DIRECT + INDIRECT inference schema")
    print("PASS: statement-type filters")
    print("PASS: DIRECT + INDIRECT graph nodes")
    print("PASS: graph construction")


if __name__ == "__main__":
    main()
