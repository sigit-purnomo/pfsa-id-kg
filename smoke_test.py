from __future__ import annotations

import json
from pathlib import Path

import torch

from pfsa_inference import (
    ENTITY2ID,
    ENTITY_LABELS,
    REL_LABELS,
    STATEMENT2ID,
    STATEMENT_LABELS,
    ConstrainedLinearChainCRF,
    build_nx_kg,
    events_to_dataframe,
    graph_edges_dataframe,
    rdf_bytes,
)


def main():
    assert "U-ROLE" in ENTITY2ID
    assert "L-AFFILIATION" in ENTITY2ID
    assert "U-STATEMENT_INDIRECT" in STATEMENT2ID
    assert "ISSUE" not in " ".join(ENTITY_LABELS)
    assert "ATTRIBUTED_TO" in REL_LABELS

    crf = ConstrainedLinearChainCRF(STATEMENT_LABELS)
    emissions = torch.zeros(1, 3, len(STATEMENT_LABELS))
    emissions[0, 0, STATEMENT2ID["I-STATEMENT_INDIRECT"]] = 20.0
    emissions[0, 0, STATEMENT2ID["U-STATEMENT_INDIRECT"]] = 10.0
    mask = torch.tensor([[True, True, True]])
    path = crf.decode(emissions, mask, default_id=STATEMENT2ID["O"])[0]
    assert path[0] != STATEMENT2ID["I-STATEMENT_INDIRECT"], "CRF allowed an illegal BILUO start"

    events = [{
        "event_id": "evt_1",
        "doc_id": "doc_1",
        "statement_type": "INDIRECT",
        "statement": {"text": "Kebijakan perlu diperbaiki.", "start": 30, "end": 58, "confidence": .93},
        "cue": {"text": "mengatakan", "start": 18, "end": 28, "label": "CUE", "confidence": .95},
        "speaker": {"text": "Raka Pratama", "canonical": "Raka Pratama", "start": 0, "end": 12, "label": "PERSON", "mention_label": "PERSON", "relation_confidence": .90},
        "role": {"text": "Ketua", "start": 13, "end": 18, "label": "ROLE", "relation_confidence": .88},
        "affiliation": {"text": "Forum Digital", "start": 60, "end": 73, "label": "AFFILIATION", "relation_confidence": .84},
        "datetime": None,
        "location": None,
        "utterance_event": None,
        "issue": {"text": "Tata kelola digital menjadi perhatian publik.", "start": 80, "end": 124, "label": "ISSUE", "confidence": .91, "relation_confidence": .91},
    }]

    graph = build_nx_kg(events)
    rels = {d.get("relation") for *_x, d in graph.edges(data=True)}
    assert "ATTRIBUTED_TO" in rels
    assert "ABOUT_ISSUE" in rels
    assert not events_to_dataframe(events).empty
    assert not graph_edges_dataframe(graph).empty
    assert rdf_bytes(events, "turtle")

    print("PFSA-ID clean Streamlit deployment smoke test: PASS")


if __name__ == "__main__":
    main()
