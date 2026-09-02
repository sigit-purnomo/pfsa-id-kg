from __future__ import annotations

import gc
import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

APP_INFERENCE_SCHEMA_VERSION = 5

STATEMENT_ENTITY_TYPES = ["STATEMENT_DIRECT", "STATEMENT_INDIRECT"]
ENTITY_CUE_TYPES = [
    "PERSON", "PERSONCOREF", "ROLE", "AFFILIATION", "DATETIME",
    "LOCATION", "EVENT", "CUE", "CUECOREF",
]
STATEMENT_LABELS = ["O"] + [f"{p}-{t}" for t in STATEMENT_ENTITY_TYPES for p in ("B", "I", "L", "U")]
ENTITY_LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_CUE_TYPES for p in ("B", "I", "L", "U")]
STATEMENT2ID = {x: i for i, x in enumerate(STATEMENT_LABELS)}
ID2STATEMENT = {i: x for x, i in STATEMENT2ID.items()}
ENTITY2ID = {x: i for i, x in enumerate(ENTITY_LABELS)}
ID2ENTITY = {i: x for x, i in ENTITY2ID.items()}

REL_LABELS = ["NONE", "ATTRIBUTED_TO", "HAS_ROLE", "AFFILIATED_WITH", "AT_TIME", "AT_LOCATION", "AT_EVENT"]
REL2ID = {x: i for i, x in enumerate(REL_LABELS)}
ID2REL = {i: x for x, i in REL2ID.items()}
KG_REL_LABELS = REL_LABELS + ["ABOUT_ISSUE"]

PRONOUNS = {"ia", "dia", "beliau", "dirinya", "mereka", "keduanya", "pihaknya"}
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])(?:["”’\)\]]*)\s+(?=[A-Z0-9“”"\(\[])')


@dataclass(frozen=True)
class InferenceSettings:
    max_length: int = 512
    stride: int = 128
    relation_threshold: float = 0.50
    speaker_max_distance: int = 500
    profile_max_distance: int = 260
    context_arg_max_distance: int = 500
    cue_max_distance: int = 160


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: Any
    manifest: dict
    bundle_manifest: dict
    checkpoint_sha256: str
    model_id: str
    resolved_revision: Optional[str]
    artifact_dir: Path
    settings: InferenceSettings
    device: torch.device


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_id(*parts: Any, prefix: str = "id") -> str:
    raw = "||".join(map(str, parts)).encode("utf-8")
    return f"{prefix}_{hashlib.sha1(raw).hexdigest()[:12]}"


def normalize_doc_id(value: Any, text: str) -> str:
    candidate = str(value or "").strip()
    if candidate and candidate.lower() != "nan":
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate).strip("_")
        if safe:
            return safe[:120]
    return stable_id(hashlib.sha1(text.encode("utf-8")).hexdigest(), prefix="doc")


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def distance_between(a0: int, a1: int, b0: int, b1: int) -> int:
    if overlaps(a0, a1, b0, b1):
        return 0
    return b0 - a1 if a1 <= b0 else a0 - b1


def sentence_spans(text: str) -> List[dict]:
    spans, start = [], 0
    for m in _SENT_BOUNDARY.finditer(text):
        end = m.start()
        if end > start:
            spans.append({"start": start, "end": end, "text": text[start:end]})
        start = m.end()
    if start < len(text):
        spans.append({"start": start, "end": len(text), "text": text[start:]})

    out = []
    for s in spans:
        raw = s["text"]
        left = len(raw) - len(raw.lstrip())
        right = len(raw) - len(raw.rstrip())
        a = s["start"] + left
        b = s["end"] - right
        if b > a:
            out.append({"start": a, "end": b, "text": text[a:b]})
    return out


def span_fully_inside_window(offsets: List[Tuple[int, int]], start: int, end: int) -> bool:
    visible = [(a, b) for a, b in offsets if b > a]
    if not visible:
        return False
    return min(a for a, _ in visible) <= int(start) and int(end) <= max(b for _, b in visible)


def token_range_for_char_span(offsets, start, end, require_full: bool = True):
    if require_full and not span_fully_inside_window(offsets, start, end):
        return None
    ids = [i for i, (a, b) in enumerate(offsets) if b > a and overlaps(a, b, int(start), int(end))]
    return None if not ids else (min(ids), max(ids))


def issue_sentence_candidates_for_window(text: str, offsets: List[Tuple[int, int]]) -> List[tuple]:
    rows = []
    for sent in sentence_spans(text):
        tr = token_range_for_char_span(offsets, sent["start"], sent["end"], require_full=True)
        if tr is not None:
            rows.append((tr[0], tr[1], 0, int(sent["start"]), int(sent["end"])))
    return rows


def _parse_biluo(label: str):
    if label == "O":
        return "O", None
    return label.split("-", 1)


def biluo_constraints(labels):
    n = len(labels)
    allowed = torch.zeros(n, n, dtype=torch.bool)
    start = torch.zeros(n, dtype=torch.bool)
    end = torch.zeros(n, dtype=torch.bool)
    for i, li in enumerate(labels):
        pi, ti = _parse_biluo(li)
        start[i] = pi in {"O", "B", "U"}
        end[i] = pi in {"O", "L", "U"}
        for j, lj in enumerate(labels):
            pj, tj = _parse_biluo(lj)
            if pi in {"O", "L", "U"}:
                ok = pj in {"O", "B", "U"}
            elif pi in {"B", "I"}:
                ok = pj in {"I", "L"} and tj == ti
            else:
                ok = False
            allowed[i, j] = ok
    return allowed, start, end


class ConstrainedLinearChainCRF(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.labels = list(labels)
        n = len(labels)
        self.transitions = nn.Parameter(torch.zeros(n, n))
        self.start_transitions = nn.Parameter(torch.zeros(n))
        self.end_transitions = nn.Parameter(torch.zeros(n))
        a, s, e = biluo_constraints(labels)
        self.register_buffer("allowed_transitions", a)
        self.register_buffer("allowed_start", s)
        self.register_buffer("allowed_end", e)

    def _masked_params(self):
        neg = -1e4
        return (
            self.transitions.masked_fill(~self.allowed_transitions, neg),
            self.start_transitions.masked_fill(~self.allowed_start, neg),
            self.end_transitions.masked_fill(~self.allowed_end, neg),
        )

    @staticmethod
    def _segments(mask1d):
        idx = torch.nonzero(mask1d, as_tuple=False).flatten().tolist()
        if not idx:
            return []
        segs, a, prev = [], idx[0], idx[0]
        for x in idx[1:]:
            if x != prev + 1:
                segs.append((a, prev + 1))
                a = x
            prev = x
        segs.append((a, prev + 1))
        return segs

    def _decode_segment(self, emit):
        tr, st, en = self._masked_params()
        score = st + emit[0]
        history = []
        for i in range(1, emit.size(0)):
            nxt = score.unsqueeze(1) + tr
            best_score, best_prev = nxt.max(dim=0)
            score = best_score + emit[i]
            history.append(best_prev)
        last = int((score + en).argmax())
        path = [last]
        for bp in reversed(history):
            last = int(bp[last])
            path.append(last)
        return list(reversed(path))

    def decode(self, emissions, mask, default_id=0):
        out = []
        for b in range(emissions.size(0)):
            seq = [default_id] * emissions.size(1)
            for a, z in self._segments(mask[b]):
                seq[a:z] = self._decode_segment(emissions[b, a:z])
            out.append(seq)
        return out


class DocumentStatementStudent(nn.Module):
    """Deployment architecture matching the clean dual-BILUO-CRF student model."""

    def __init__(self, encoder_config):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_config(encoder_config)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.statement_head = nn.Linear(h, len(STATEMENT_LABELS))
        self.entity_head = nn.Linear(h, len(ENTITY_LABELS))
        self.statement_crf = ConstrainedLinearChainCRF(STATEMENT_LABELS)
        self.entity_crf = ConstrainedLinearChainCRF(ENTITY_LABELS)
        self.issue_head = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(h // 2, 1))
        self.rel_head = nn.Sequential(nn.Linear(h * 4, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, len(REL_LABELS)))
        self.rel_loss_fn = nn.CrossEntropyLoss()
        self.register_buffer("statement_class_weights", torch.ones(len(STATEMENT_LABELS)))
        self.register_buffer("entity_class_weights", torch.ones(len(ENTITY_LABELS)))

    @staticmethod
    def span_pool(hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return hidden[start:end + 1].mean(dim=0)

    def forward(self, input_ids, attention_mask, issue_candidates=None):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(enc.last_hidden_state)
        statement_logits = self.statement_head(hidden)
        entity_logits = self.entity_head(hidden)

        issue_logits, issue_meta = [], []
        if issue_candidates is not None:
            for b, cands in enumerate(issue_candidates):
                for s0, s1, _target, *meta in cands:
                    vec = self.span_pool(hidden[b], s0, s1)
                    issue_logits.append(self.issue_head(vec).squeeze(-1))
                    issue_meta.append((b, s0, s1, *meta))

        return {
            "statement_logits": statement_logits,
            "entity_logits": entity_logits,
            "issue_logits": torch.stack(issue_logits) if issue_logits else None,
            "issue_meta": issue_meta,
            "hidden": hidden,
        }


def _biluo_spans_from_ids(offsets, ids, id2label):
    spans, i = [], 0
    while i < len(ids):
        lab = id2label.get(int(ids[i]), "O")
        p, t = _parse_biluo(lab)
        if p == "U" and offsets[i][1] > offsets[i][0]:
            spans.append((int(offsets[i][0]), int(offsets[i][1]), t))
            i += 1
            continue
        if p == "B":
            j = i + 1
            while j < len(ids):
                pj, tj = _parse_biluo(id2label.get(int(ids[j]), "O"))
                if tj == t and pj == "L":
                    spans.append((int(offsets[i][0]), int(offsets[j][1]), t))
                    j += 1
                    break
                if not (tj == t and pj == "I"):
                    break
                j += 1
            i = max(i + 1, j)
            continue
        i += 1
    return spans


def _decode_biluo_window(offsets, pred_ids, id2label, token_probs=None):
    spans = []
    for a, b, typ in _biluo_spans_from_ids(offsets, pred_ids, id2label):
        ids = [i for i, (x, y) in enumerate(offsets) if y > x and overlaps(x, y, a, b)]
        conf = float(np.mean([token_probs[i] for i in ids])) if token_probs is not None and ids else 0.0
        spans.append({"type": typ, "start": a, "end": b, "confidence": conf})
    return spans


def _dedupe_predicted_spans(spans):
    best = {}
    for s in spans:
        key = (s["type"], s["start"], s["end"])
        if key not in best or s["confidence"] > best[key]["confidence"]:
            best[key] = s
    return sorted(best.values(), key=lambda x: (x["start"], x["end"], x["type"]))


def _sentence_for_offset(text, pos):
    return next((s for s in sentence_spans(text) if s["start"] <= pos < s["end"]), None)


def predict_statement_events(text: str, bundle: ModelBundle, doc_id: str, relation_threshold: Optional[float] = None):
    model, tokenizer = bundle.model, bundle.tokenizer
    settings = bundle.settings
    relation_threshold = settings.relation_threshold if relation_threshold is None else float(relation_threshold)
    device = bundle.device
    model.eval()

    enc = tokenizer(
        text,
        truncation=True,
        max_length=settings.max_length,
        stride=settings.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=None,
    )

    all_spans, window_cache, issue_scores = [], [], {}
    with torch.no_grad():
        for wi in range(len(enc["input_ids"])):
            ids = torch.tensor([enc["input_ids"][wi]], dtype=torch.long, device=device)
            mask = torch.tensor([enc["attention_mask"][wi]], dtype=torch.long, device=device)
            offsets = [tuple(x) for x in enc["offset_mapping"][wi]]
            issue_cands = issue_sentence_candidates_for_window(text, offsets)
            out = model(input_ids=ids, attention_mask=mask, issue_candidates=[issue_cands])

            valid = torch.tensor([[bool(y > x) for x, y in offsets]], dtype=torch.bool, device=device) & mask.bool()
            st_ids = model.statement_crf.decode(out["statement_logits"], valid, STATEMENT2ID["O"])[0]
            en_ids = model.entity_crf.decode(out["entity_logits"], valid, ENTITY2ID["O"])[0]
            st_prob = torch.softmax(out["statement_logits"][0], dim=-1).max(-1).values.cpu().tolist()
            en_prob = torch.softmax(out["entity_logits"][0], dim=-1).max(-1).values.cpu().tolist()

            all_spans.extend(_decode_biluo_window(offsets, st_ids, ID2STATEMENT, st_prob))
            all_spans.extend(_decode_biluo_window(offsets, en_ids, ID2ENTITY, en_prob))

            if out["issue_logits"] is not None:
                probs = torch.sigmoid(out["issue_logits"]).cpu().tolist()
                for prob, meta in zip(probs, out["issue_meta"]):
                    _, _, _, a, b = meta
                    issue_scores[(int(a), int(b))] = max(float(prob), issue_scores.get((int(a), int(b)), -1.0))

            window_cache.append({"offsets": offsets, "hidden": out["hidden"]})

    spans = _dedupe_predicted_spans(all_spans)
    for s in spans:
        s["text"] = text[s["start"]:s["end"]]

    statements = [s for s in spans if s["type"] in STATEMENT_ENTITY_TYPES]
    people = [s for s in spans if s["type"] in {"PERSON", "PERSONCOREF"}]
    roles = [s for s in spans if s["type"] == "ROLE"]
    affiliations = [s for s in spans if s["type"] == "AFFILIATION"]
    context_args = [s for s in spans if s["type"] in {"DATETIME", "LOCATION", "EVENT"}]
    cues = [s for s in spans if s["type"] in {"CUE", "CUECOREF"}]
    explicit_people = [s for s in people if s["type"] == "PERSON"]

    issue_obj = None
    if issue_scores:
        (ia, ib), isc = max(issue_scores.items(), key=lambda kv: kv[1])
        issue_obj = {
            "text": text[ia:ib], "evidence": text[ia:ib], "start": ia, "end": ib,
            "label": "ISSUE", "confidence": float(isc), "relation_confidence": float(isc),
            "prediction_head": "ISSUE_SENTENCE_HEAD",
        }

    def coref_canonical(arg):
        mention = (arg.get("text") or "").strip().lower()
        prior = [p for p in explicit_people if p["end"] <= arg["start"]]
        matches = [
            p for p in prior
            if mention and mention not in PRONOUNS
            and mention in {t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ.'-]+", p.get("text", ""))}
        ]
        if matches:
            return matches[-1]["text"]
        if prior:
            return prior[-1]["text"]
        return arg.get("text")

    def relation_score(statement, argument, rel_name):
        rid, best = REL2ID[rel_name], -1.0
        for wc in window_cache:
            sr = token_range_for_char_span(wc["offsets"], statement["start"], statement["end"], True)
            ar = token_range_for_char_span(wc["offsets"], argument["start"], argument["end"], True)
            if sr is None or ar is None:
                continue
            sv = DocumentStatementStudent.span_pool(wc["hidden"][0], sr[0], sr[1])
            av = DocumentStatementStudent.span_pool(wc["hidden"][0], ar[0], ar[1])
            feat = torch.cat([sv, av, torch.abs(sv - av), sv * av], dim=-1)
            prob = torch.softmax(model.rel_head(feat), dim=-1)
            best = max(best, float(prob[rid].cpu()))
        return best

    events = []
    for si, st in enumerate(statements):
        near_cues = sorted(cues, key=lambda c: distance_between(st["start"], st["end"], c["start"], c["end"]))
        cue = near_cues[0] if near_cues and distance_between(st["start"], st["end"], near_cues[0]["start"], near_cues[0]["end"]) <= settings.cue_max_distance else None

        speaker_best = None
        for person in people:
            if distance_between(st["start"], st["end"], person["start"], person["end"]) > settings.speaker_max_distance:
                continue
            score = relation_score(st, person, "ATTRIBUTED_TO")
            if score >= relation_threshold and (speaker_best is None or score > speaker_best[0]):
                speaker_best = (score, person)

        fields = {"speaker": None}
        if speaker_best:
            score, person = speaker_best
            speaker = {
                "text": person["text"], "evidence": person["text"], "start": person["start"], "end": person["end"],
                "label": person["type"], "relation_confidence": score,
            }
            if person["type"] == "PERSON":
                speaker.update({"canonical": person["text"], "mention_label": "PERSON", "coref_resolution_provenance": "EXPLICIT_PREDICTED_PERSON"})
            else:
                speaker.update({"canonical": coref_canonical(person), "mention_label": "PERSONCOREF", "coref_resolution_provenance": "DOCUMENT_HEURISTIC_FROM_PREDICTED_PERSON"})
            fields["speaker"] = speaker

        profile_anchor = fields["speaker"] or st
        profile_sent = _sentence_for_offset(text, profile_anchor["start"])
        for candidates, field, relation in [
            (roles, "role", "HAS_ROLE"),
            (affiliations, "affiliation", "AFFILIATED_WITH"),
        ]:
            best = None
            for arg in candidates:
                if distance_between(profile_anchor["start"], profile_anchor["end"], arg["start"], arg["end"]) > settings.profile_max_distance:
                    continue
                arg_sent = _sentence_for_offset(text, arg["start"])
                if profile_sent and arg_sent and not overlaps(profile_sent["start"], profile_sent["end"], arg_sent["start"], arg_sent["end"]):
                    continue
                score = relation_score(st, arg, relation)
                if score >= relation_threshold and (best is None or score > best[0]):
                    best = (score, arg)
            fields[field] = None if best is None else {
                "text": best[1]["text"], "evidence": best[1]["text"], "start": best[1]["start"], "end": best[1]["end"],
                "label": best[1]["type"], "relation_confidence": best[0],
            }

        for typ, field, relation in [
            ("DATETIME", "datetime", "AT_TIME"),
            ("LOCATION", "location", "AT_LOCATION"),
            ("EVENT", "utterance_event", "AT_EVENT"),
        ]:
            best = None
            for arg in [x for x in context_args if x["type"] == typ]:
                if distance_between(st["start"], st["end"], arg["start"], arg["end"]) > settings.context_arg_max_distance:
                    continue
                score = relation_score(st, arg, relation)
                if score >= relation_threshold and (best is None or score > best[0]):
                    best = (score, arg)
            fields[field] = None if best is None else {
                "text": best[1]["text"], "evidence": best[1]["text"], "start": best[1]["start"], "end": best[1]["end"],
                "label": best[1]["type"], "relation_confidence": best[0],
            }

        fields["issue"] = issue_obj
        if cue is not None and cue["type"] == "CUECOREF" and fields.get("speaker"):
            fields["speaker"]["coreference_via"] = "CUECOREF"

        events.append({
            "event_id": stable_id(doc_id, st["start"], st["end"], si, "prediction", prefix="stmt"),
            "doc_id": doc_id,
            "statement_type": "DIRECT" if st["type"] == "STATEMENT_DIRECT" else "INDIRECT",
            "statement": {
                "text": st["text"], "evidence": st["text"], "start": st["start"], "end": st["end"],
                "confidence": st["confidence"],
            },
            "cue": None if cue is None else {
                "text": cue["text"], "evidence": cue["text"], "start": cue["start"], "end": cue["end"],
                "confidence": cue["confidence"], "label": cue["type"],
            },
            "span_provenance": "STUDENT_MODEL_BILUO_CRF",
            "relation_provenance": "STUDENT_MODEL_CONSTRAINED_RELATION_HEAD",
            **fields,
        })

    return events, spans


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _state_dict_from_payload(payload):
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    raise RuntimeError("Checkpoint does not contain a model state_dict.")


def _training_config(manifest: dict) -> dict:
    return manifest.get("fingerprint_payload", {}).get("training_config", {})


def inspect_artifact_dir(artifact_dir: Path, verify_checkpoint: bool = False) -> dict:
    artifact_dir = Path(artifact_dir)
    checkpoint = artifact_dir / "student_model_best.pt"
    manifest_path = artifact_dir / "student_model_manifest.json"
    tokenizer_dir = artifact_dir / "student_tokenizer"
    encoder_config_dir = artifact_dir / "student_encoder_config"
    bundle_manifest_path = artifact_dir / "deployment_bundle_manifest.json"

    problems = []
    for p, label in [(checkpoint, "student_model_best.pt"), (manifest_path, "student_model_manifest.json")]:
        if not p.exists():
            problems.append(f"Missing {label}")
    if not tokenizer_dir.exists():
        problems.append("Missing student_tokenizer/")
    if not encoder_config_dir.exists():
        problems.append("Missing student_encoder_config/")

    manifest, bundle_manifest = {}, {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("training_complete") is not True:
                problems.append("Training manifest does not mark training_complete=True")
        except Exception as e:
            problems.append(f"Invalid training manifest: {e}")

    if bundle_manifest_path.exists():
        try:
            bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            problems.append(f"Invalid deployment bundle manifest: {e}")

    checkpoint_sha = None
    if checkpoint.exists() and verify_checkpoint:
        checkpoint_sha = sha256_file(checkpoint)
        expected_sha = (manifest.get("checkpoint") or {}).get("sha256")
        if expected_sha and expected_sha != checkpoint_sha:
            problems.append("Checkpoint SHA-256 does not match training manifest")

    if checkpoint.exists() and verify_checkpoint and not problems:
        try:
            payload = _safe_torch_load(checkpoint)
            if list(payload.get("statement_labels", [])) != STATEMENT_LABELS:
                problems.append("Checkpoint statement-label schema is not the clean BILUO schema")
            if list(payload.get("entity_labels", [])) != ENTITY_LABELS:
                problems.append("Checkpoint entity/cue-label schema is not the clean BILUO schema")
            if list(payload.get("relation_labels", [])) != REL_LABELS:
                problems.append("Checkpoint relation-label schema is incompatible")
        except Exception as e:
            problems.append(f"Checkpoint could not be inspected: {e}")

    return {
        "ready": not problems,
        "artifact_dir": str(artifact_dir),
        "problems": problems,
        "training_complete": manifest.get("training_complete"),
        "training_fingerprint": manifest.get("training_fingerprint"),
        "manifest_schema_version": manifest.get("manifest_schema_version"),
        "checkpoint_sha256": checkpoint_sha,
        "architecture": _training_config(manifest).get("architecture"),
        "student_model_id": _training_config(manifest).get("student_model_id") or bundle_manifest.get("student_model_id"),
        "teacher_model_id": bundle_manifest.get("teacher_model_id"),
    }


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    d = torch.device(device)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return d


def load_model_bundle(artifact_dir: Path, relation_threshold: float = 0.50, device: str = "auto") -> ModelBundle:
    artifact_dir = Path(artifact_dir)
    audit = inspect_artifact_dir(artifact_dir, verify_checkpoint=True)
    if not audit["ready"]:
        raise RuntimeError("Model artifact is not ready: " + "; ".join(audit["problems"]))

    from transformers import AutoConfig, AutoTokenizer

    checkpoint_path = artifact_dir / "student_model_best.pt"
    manifest = json.loads((artifact_dir / "student_model_manifest.json").read_text(encoding="utf-8"))
    bundle_manifest_path = artifact_dir / "deployment_bundle_manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8")) if bundle_manifest_path.exists() else {}
    payload = _safe_torch_load(checkpoint_path)

    encoder_config = AutoConfig.from_pretrained(artifact_dir / "student_encoder_config", local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "student_tokenizer", local_files_only=True, use_fast=True, trust_remote_code=False)
    model = DocumentStatementStudent(encoder_config)
    model.load_state_dict(_state_dict_from_payload(payload), strict=True)

    dev = _resolve_device(device)
    model.to(dev).eval()

    cfg = _training_config(manifest)
    settings = InferenceSettings(
        max_length=int(cfg.get("max_length", 512)),
        stride=int(cfg.get("stride", 128)),
        relation_threshold=float(relation_threshold),
        speaker_max_distance=int(bundle_manifest.get("inference_speaker_max_distance", 500)),
        profile_max_distance=int(bundle_manifest.get("inference_profile_max_distance", 260)),
        context_arg_max_distance=int(bundle_manifest.get("inference_context_arg_max_distance", 500)),
        cue_max_distance=int(bundle_manifest.get("inference_cue_max_distance", 160)),
    )

    model_id = payload.get("student_model_id") or cfg.get("student_model_id") or bundle_manifest.get("student_model_id") or "indobenchmark/indobert-base-p1"
    revision = payload.get("student_resolved_revision") or cfg.get("student_resolved_revision") or bundle_manifest.get("student_resolved_revision")

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        manifest=manifest,
        bundle_manifest=bundle_manifest,
        checkpoint_sha256=sha256_file(checkpoint_path),
        model_id=str(model_id),
        resolved_revision=None if revision in (None, "") else str(revision),
        artifact_dir=artifact_dir,
        settings=settings,
        device=dev,
    )


def release_model_bundle(bundle: Optional[ModelBundle]) -> None:
    if bundle is not None:
        try:
            bundle.model.to("cpu")
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def list_runtime_devices() -> List[dict]:
    rows = [{"device": "cpu", "name": "CPU", "free_gb": None, "total_gb": None}]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
                rows.append({
                    "device": f"cuda:{i}",
                    "name": torch.cuda.get_device_name(i),
                    "free_gb": free / 1024**3,
                    "total_gb": total / 1024**3,
                })
            except Exception:
                rows.append({"device": f"cuda:{i}", "name": torch.cuda.get_device_name(i), "free_gb": None, "total_gb": None})
    return rows


def inference_cache_key(bundle: ModelBundle, text: str, doc_id: str, relation_threshold: float) -> str:
    payload = {
        "schema": APP_INFERENCE_SCHEMA_VERSION,
        "checkpoint": bundle.checkpoint_sha256,
        "doc_id": doc_id,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "max_length": bundle.settings.max_length,
        "stride": bundle.settings.stride,
        "relation_threshold": float(relation_threshold),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def infer_with_disk_cache(bundle: ModelBundle, text: str, doc_id: str, cache_dir: Path, relation_threshold: float, use_cache: bool = True):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = inference_cache_key(bundle, text, doc_id, relation_threshold)
    path = cache_dir / f"{key}.json"
    if use_cache and path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj["events"], obj["spans"], True

    events, spans = predict_statement_events(text, bundle, doc_id, relation_threshold)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"events": events, "spans": spans}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return events, spans, False


def _slug(text: str) -> str:
    x = re.sub(r"[^\w]+", "_", str(text).strip().lower(), flags=re.UNICODE).strip("_")
    return x[:120] or "unknown"


def _add_entity_node(graph, prefix, obj, node_type):
    if not obj:
        return None
    label = obj.get("canonical") or obj.get("text") or obj.get("evidence")
    if not label:
        return None
    node_id = f"{prefix}:{_slug(label)}"
    graph.add_node(node_id, node_type=node_type, label=str(label))
    return node_id


def build_nx_kg(events: List[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for event in events:
        doc_node = f"article:{event['doc_id']}"
        graph.add_node(doc_node, node_type="Article", label=event["doc_id"])

        statement = event["statement"]
        event_node = f"statement_event:{event['event_id']}"
        graph.add_node(
            event_node,
            node_type="StatementEvent",
            label=statement.get("text", ""),
            content=statement.get("text", ""),
            statement_type=event.get("statement_type"),
            confidence=float(statement.get("confidence", 0.0)),
            evidence_start=int(statement.get("start", -1)),
            evidence_end=int(statement.get("end", -1)),
        )
        graph.add_edge(event_node, doc_node, relation="SOURCE_ARTICLE")

        mapping = [
            ("speaker", "person", "Person", "ATTRIBUTED_TO"),
            ("role", "role", "Role", "HAS_ROLE"),
            ("affiliation", "org", "Organization", "AFFILIATED_WITH"),
            ("datetime", "time", "DateTime", "AT_TIME"),
            ("location", "place", "Location", "AT_LOCATION"),
            ("utterance_event", "event", "UtteranceEvent", "AT_EVENT"),
            ("issue", "issue", "Issue", "ABOUT_ISSUE"),
        ]
        for field, prefix, node_type, relation in mapping:
            obj = event.get(field)
            target = _add_entity_node(graph, prefix, obj, node_type)
            if not target:
                continue
            attrs = {"relation": relation}
            if obj.get("relation_confidence") is not None:
                attrs["relation_confidence"] = float(obj["relation_confidence"])
            if obj.get("label") is not None:
                attrs["source_span_label"] = str(obj["label"])
            if obj.get("mention_label") is not None:
                attrs["source_mention_label"] = str(obj["mention_label"])
            if obj.get("coreference_via") is not None:
                attrs["coreference_via"] = str(obj["coreference_via"])
            if obj.get("start") is not None:
                attrs.update({
                    "evidence_start": int(obj["start"]),
                    "evidence_end": int(obj["end"]),
                    "evidence_text": obj.get("evidence") or obj.get("text") or "",
                })
            graph.add_edge(event_node, target, **attrs)
    return graph


def events_to_dataframe(events: List[dict]) -> pd.DataFrame:
    rows = []
    for e in events:
        sp = e.get("speaker") or {}
        st = e.get("statement") or {}
        cue = e.get("cue") or {}
        rows.append({
            "event_id": e.get("event_id"),
            "statement_type": e.get("statement_type"),
            "statement": st.get("text"),
            "statement_confidence": st.get("confidence"),
            "cue": cue.get("text"),
            "cue_label": cue.get("label"),
            "speaker": sp.get("canonical") or sp.get("text"),
            "speaker_mention_label": sp.get("mention_label") or sp.get("label"),
            "speaker_relation_confidence": sp.get("relation_confidence"),
            "role": (e.get("role") or {}).get("text"),
            "affiliation": (e.get("affiliation") or {}).get("text"),
            "datetime": (e.get("datetime") or {}).get("text"),
            "location": (e.get("location") or {}).get("text"),
            "utterance_event": (e.get("utterance_event") or {}).get("text"),
            "issue": (e.get("issue") or {}).get("text"),
            "issue_confidence": (e.get("issue") or {}).get("confidence"),
        })
    return pd.DataFrame(rows)


def spans_to_dataframe(spans: List[dict], doc_id: str = "document") -> pd.DataFrame:
    return pd.DataFrame([
        {"doc_id": doc_id, "type": s.get("type"), "text": s.get("text"), "start": s.get("start"), "end": s.get("end"), "confidence": s.get("confidence")}
        for s in spans
    ])


def relation_rows_for_event(event: dict) -> List[dict]:
    rows = []
    mapping = [
        ("speaker", "ATTRIBUTED_TO"), ("role", "HAS_ROLE"), ("affiliation", "AFFILIATED_WITH"),
        ("datetime", "AT_TIME"), ("location", "AT_LOCATION"), ("utterance_event", "AT_EVENT"), ("issue", "ABOUT_ISSUE"),
    ]
    for field, relation in mapping:
        obj = event.get(field)
        if obj:
            rows.append({
                "relation": relation,
                "value": obj.get("canonical") or obj.get("text"),
                "confidence": obj.get("relation_confidence"),
                "span_label": obj.get("label") or obj.get("mention_label"),
            })
    return rows


def graph_edges_dataframe(graph: nx.MultiDiGraph) -> pd.DataFrame:
    rows = []
    for source, target, _key, attrs in graph.edges(keys=True, data=True):
        rows.append({"source": source, "relation": attrs.get("relation"), "target": target, "confidence": attrs.get("relation_confidence"), "evidence": attrs.get("evidence_text")})
    return pd.DataFrame(rows)


def events_to_jsonl_bytes(events: List[dict]) -> bytes:
    return ("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in events)).encode("utf-8")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def graphml_bytes(graph: nx.MultiDiGraph) -> bytes:
    bio = io.BytesIO()
    nx.write_graphml(graph, bio, encoding="utf-8")
    return bio.getvalue()


PFSA = Namespace("https://example.org/pfsa/")


def _uri(kind: str, value: str) -> URIRef:
    return URIRef(str(PFSA) + kind + "/" + _slug(value))


def build_rdf_kg(events: List[dict]) -> Graph:
    g = Graph()
    g.bind("pfsa", PFSA)
    mapping = [
        ("speaker", "person", PFSA.Person, PFSA.attributedTo),
        ("role", "role", PFSA.Role, PFSA.hasRole),
        ("affiliation", "organization", PFSA.Organization, PFSA.affiliatedWith),
        ("datetime", "datetime", PFSA.DateTime, PFSA.atTime),
        ("location", "location", PFSA.Location, PFSA.atLocation),
        ("utterance_event", "utterance-event", PFSA.UtteranceEvent, PFSA.atEvent),
        ("issue", "issue", PFSA.Issue, PFSA.aboutIssue),
    ]
    for e in events:
        ev = _uri("statement-event", e["event_id"])
        art = _uri("article", e["doc_id"])
        g.add((ev, RDF.type, PFSA.StatementEvent))
        g.add((art, RDF.type, PFSA.Article))
        g.add((ev, PFSA.sourceArticle, art))
        g.add((ev, PFSA.statementType, Literal(e.get("statement_type"))))
        g.add((ev, PFSA.content, Literal((e.get("statement") or {}).get("text", ""), lang="id")))
        for field, kind, rdf_type, predicate in mapping:
            obj = e.get(field)
            if not obj:
                continue
            label = obj.get("canonical") or obj.get("text") or obj.get("evidence")
            if not label:
                continue
            target = _uri(kind, label)
            g.add((target, RDF.type, rdf_type))
            g.add((target, PFSA.label, Literal(label, lang="id")))
            g.add((ev, predicate, target))
            if obj.get("relation_confidence") is not None:
                g.add((target, PFSA.relationConfidence, Literal(float(obj["relation_confidence"]), datatype=XSD.double)))
    return g


def rdf_bytes(events: List[dict], fmt: str = "turtle") -> bytes:
    data = build_rdf_kg(events).serialize(format=fmt)
    return data.encode("utf-8") if isinstance(data, str) else data
