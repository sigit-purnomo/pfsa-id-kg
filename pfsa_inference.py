from __future__ import annotations

import gc
import hashlib
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


# ---------------------------------------------------------------------------
# V14 deployment schema
# ---------------------------------------------------------------------------

APP_INFERENCE_SCHEMA_VERSION = 4

TOKEN_ENTITY_TYPES = [
    "STATEMENT_DIRECT",
    "STATEMENT_INDIRECT",
    "PERSON",
    "PERSONCOREF",
    "ROLE",
    "AFFILIATION",
    "DATETIME",
    "LOCATION",
    "EVENT",
    "ISSUE",
    "CUE",
    "CUECOREF",
]
TOKEN_LABELS = ["O"] + [f"{p}-{t}" for t in TOKEN_ENTITY_TYPES for p in ("B", "I")]
TOKEN2ID = {x: i for i, x in enumerate(TOKEN_LABELS)}
ID2TOKEN = {i: x for x, i in TOKEN2ID.items()}

REL_LABELS = [
    "NONE",
    "ATTRIBUTED_TO",
    "HAS_ROLE",
    "AFFILIATED_WITH",
    "AT_TIME",
    "AT_LOCATION",
    "AT_EVENT",
    "ABOUT_ISSUE",
]
REL2ID = {x: i for i, x in enumerate(REL_LABELS)}
ID2REL = {i: x for x, i in REL2ID.items()}

PRONOUNS = {
    "ia",
    "dia",
    "beliau",
    "mereka",
    "dirinya",
    "pihaknya",
    "keduanya",
}


@dataclass(frozen=True)
class InferenceSettings:
    max_length: int = 512
    stride: int = 128
    relation_threshold: float = 0.50
    max_relation_char_distance: int = 1800
    max_cue_char_distance: int = 160


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: Any
    manifest: dict
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


def normalize_doc_id(value: Any, text: str, index: Optional[int] = None) -> str:
    if value is not None:
        candidate = str(value).strip()
        if candidate and candidate.lower() != "nan":
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate).strip("_")
            if safe:
                return safe[:120]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return stable_id(index if index is not None else "document", digest, prefix="doc")


def overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def distance_between(a0: int, a1: int, b0: int, b1: int) -> int:
    if overlaps(a0, a1, b0, b1):
        return 0
    if a1 <= b0:
        return b0 - a1
    return a0 - b1


def token_range_for_char_span(
    offsets: List[Tuple[int, int]],
    start: int,
    end: int,
) -> Optional[Tuple[int, int]]:
    ids = [i for i, (a, b) in enumerate(offsets) if b > a and overlaps(a, b, start, end)]
    if not ids:
        return None
    return min(ids), max(ids)


class DocumentStatementStudent(nn.Module):
    """Deployment architecture matching the V14 notebook."""

    def __init__(self, encoder_config, n_token_labels: int, n_rel_labels: int):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_config(encoder_config)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.token_head = nn.Linear(hidden_size, n_token_labels)
        self.rel_head = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, n_rel_labels),
        )
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.rel_loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def span_pool(hidden: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return hidden[start : end + 1].mean(dim=0)

    def forward(self, input_ids, attention_mask, labels=None, relation_pairs=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(out.last_hidden_state)
        token_logits = self.token_head(hidden)

        token_loss = None
        if labels is not None:
            token_loss = self.token_loss_fn(
                token_logits.view(-1, token_logits.size(-1)),
                labels.view(-1),
            )

        rel_logits_all, rel_targets = [], []
        if relation_pairs is not None:
            for batch_idx, pairs in enumerate(relation_pairs):
                for s0, s1, a0, a1, rel_id in pairs:
                    svec = self.span_pool(hidden[batch_idx], s0, s1)
                    avec = self.span_pool(hidden[batch_idx], a0, a1)
                    feat = torch.cat(
                        [svec, avec, torch.abs(svec - avec), svec * avec],
                        dim=-1,
                    )
                    rel_logits_all.append(self.rel_head(feat))
                    rel_targets.append(rel_id)

        rel_logits = torch.stack(rel_logits_all) if rel_logits_all else None
        rel_loss = None
        if rel_logits_all and rel_targets:
            target_tensor = torch.tensor(
                rel_targets,
                dtype=torch.long,
                device=rel_logits.device,
            )
            rel_loss = self.rel_loss_fn(rel_logits, target_tensor)

        return {
            "token_logits": token_logits,
            "relation_logits": rel_logits,
            "token_loss": token_loss,
            "relation_loss": rel_loss,
        }


def _safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # The artifact is expected to be produced by the user's own V14 notebook.
        return torch.load(path, map_location="cpu", weights_only=False)


def _state_dict_from_payload(payload: Any) -> dict:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    if isinstance(payload, dict) and payload and all(isinstance(k, str) for k in payload):
        return payload
    raise RuntimeError("Unsupported checkpoint format; no model state_dict was found.")


def _training_config(manifest: dict) -> dict:
    return (
        manifest.get("fingerprint_payload", {})
        .get("training_config", {})
    )


def _model_id_from_sources(manifest: dict, payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("student_model_id"):
        return str(payload["student_model_id"])
    return str(_training_config(manifest).get("student_model_id", "indobenchmark/indobert-base-p1"))


def _resolved_revision(manifest: dict, payload: Any) -> Optional[str]:
    if isinstance(payload, dict) and payload.get("student_resolved_revision"):
        return str(payload["student_resolved_revision"])
    value = _training_config(manifest).get("student_resolved_revision")
    return None if value in (None, "") else str(value)


def _settings_from_manifest(manifest: dict, relation_threshold: float) -> InferenceSettings:
    cfg = _training_config(manifest)
    return InferenceSettings(
        max_length=int(cfg.get("max_length", 512)),
        stride=int(cfg.get("stride", 128)),
        relation_threshold=float(relation_threshold),
        max_relation_char_distance=1800,
        max_cue_char_distance=160,
    )


def inspect_artifact_dir(artifact_dir: Path, verify_checkpoint: bool = False) -> dict:
    artifact_dir = Path(artifact_dir)
    checkpoint = artifact_dir / "student_model_best.pt"
    manifest_path = artifact_dir / "student_model_manifest.json"
    tokenizer_dir = artifact_dir / "student_tokenizer"
    encoder_config_dir = artifact_dir / "student_encoder_config"
    bundle_manifest_path = artifact_dir / "deployment_bundle_manifest.json"

    result = {
        "artifact_dir": str(artifact_dir),
        "checkpoint_exists": checkpoint.exists(),
        "manifest_exists": manifest_path.exists(),
        "tokenizer_exists": tokenizer_dir.exists(),
        "encoder_config_exists": encoder_config_dir.exists(),
        "deployment_bundle_manifest_exists": bundle_manifest_path.exists(),
        "ready": False,
        "problems": [],
    }

    if not checkpoint.exists():
        result["problems"].append("student_model_best.pt not found")
    if not manifest_path.exists():
        result["problems"].append("student_model_manifest.json not found")
    if not tokenizer_dir.exists():
        result["problems"].append("student_tokenizer/ not found")

    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result["training_complete"] = manifest.get("training_complete")
            result["training_fingerprint"] = manifest.get("training_fingerprint")
            result["manifest_schema_version"] = manifest.get("manifest_schema_version")
            result["best_epoch"] = manifest.get("best_epoch")
            result["best_validation_loss"] = manifest.get("best_validation_loss")
            cfg = _training_config(manifest)
            result["student_model_id"] = cfg.get("student_model_id")
            result["student_resolved_revision"] = cfg.get("student_resolved_revision")
            result["max_length"] = cfg.get("max_length")
            result["stride"] = cfg.get("stride")
            if manifest.get("training_complete") is not True:
                result["problems"].append("training manifest does not mark training_complete=True")
        except Exception as exc:
            result["problems"].append(f"student_model_manifest.json could not be read: {exc}")

    if checkpoint.exists() and verify_checkpoint:
        try:
            actual_sha = sha256_file(checkpoint)
            result["checkpoint_sha256"] = actual_sha
            if manifest:
                expected_sha = (manifest.get("checkpoint") or {}).get("sha256")
                if expected_sha and expected_sha != actual_sha:
                    result["problems"].append("checkpoint SHA-256 differs from training manifest")
            payload = _safe_torch_load(checkpoint)
            if isinstance(payload, dict):
                saved_token_labels = payload.get("token_labels")
                saved_rel_labels = payload.get("relation_labels")
                if saved_token_labels is not None and list(saved_token_labels) != TOKEN_LABELS:
                    result["problems"].append(
                        "checkpoint token-label schema is not V14 compatible "
                        "(PERSONCOREF/CUECOREF/ISSUE expected)"
                    )
                if saved_rel_labels is not None and list(saved_rel_labels) != REL_LABELS:
                    result["problems"].append(
                        "checkpoint relation-label schema is not V14 compatible "
                        "(ABOUT_ISSUE expected)"
                    )
                result["checkpoint_schema_ok"] = (
                    saved_token_labels is None or list(saved_token_labels) == TOKEN_LABELS
                ) and (
                    saved_rel_labels is None or list(saved_rel_labels) == REL_LABELS
                )
        except Exception as exc:
            result["problems"].append(f"checkpoint metadata could not be inspected: {exc}")
    elif checkpoint.exists():
        result["checkpoint_sha256_expected"] = (
            (manifest.get("checkpoint") or {}).get("sha256")
            if manifest
            else None
        )

    if bundle_manifest_path.exists():
        try:
            deployment_manifest = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
            result["deployment_bundle_version"] = deployment_manifest.get("bundle_version")
            result["compatible_pipeline"] = deployment_manifest.get("compatible_pipeline")
            bundle_token_labels = deployment_manifest.get("v14_token_labels")
            bundle_rel_labels = deployment_manifest.get("v14_relation_labels")
            if bundle_token_labels is not None and list(bundle_token_labels) != TOKEN_LABELS:
                result["problems"].append("deployment bundle token schema is not V14 compatible")
            if bundle_rel_labels is not None and list(bundle_rel_labels) != REL_LABELS:
                result["problems"].append("deployment bundle relation schema is not V14 compatible")
        except Exception as exc:
            result["problems"].append(f"deployment_bundle_manifest.json could not be read: {exc}")

    result["ready"] = len(result["problems"]) == 0
    return result


def list_runtime_devices() -> List[dict]:
    rows = [{"device": "cpu", "name": "CPU", "total_gb": None, "free_gb": None}]
    if not torch.cuda.is_available():
        return rows
    for idx in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(idx)
            rows.append(
                {
                    "device": f"cuda:{idx}",
                    "name": torch.cuda.get_device_name(idx),
                    "total_gb": round(total / (1024 ** 3), 2),
                    "free_gb": round(free / (1024 ** 3), 2),
                }
            )
        except Exception:
            rows.append(
                {
                    "device": f"cuda:{idx}",
                    "name": torch.cuda.get_device_name(idx),
                    "total_gb": None,
                    "free_gb": None,
                }
            )
    return rows


def release_model_bundle(bundle: Optional[ModelBundle]) -> None:
    if bundle is None:
        return
    try:
        bundle.model.to("cpu")
    except Exception:
        pass
    try:
        del bundle.model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_bundle(
    artifact_dir: Path,
    relation_threshold: float = 0.50,
    device: Optional[str] = None,
) -> ModelBundle:
    artifact_dir = Path(artifact_dir)
    checkpoint_path = artifact_dir / "student_model_best.pt"
    manifest_path = artifact_dir / "student_model_manifest.json"
    tokenizer_dir = artifact_dir / "student_tokenizer"
    encoder_config_dir = artifact_dir / "student_encoder_config"

    status = inspect_artifact_dir(artifact_dir, verify_checkpoint=True)
    if not status["ready"]:
        raise RuntimeError("Artifact bundle is not ready: " + "; ".join(status["problems"]))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint_path)
    payload = _safe_torch_load(checkpoint_path)

    saved_token_labels = payload.get("token_labels") if isinstance(payload, dict) else None
    saved_rel_labels = payload.get("relation_labels") if isinstance(payload, dict) else None
    if saved_token_labels is not None and list(saved_token_labels) != TOKEN_LABELS:
        raise RuntimeError("Checkpoint token-label schema does not match the V14 application schema.")
    if saved_rel_labels is not None and list(saved_rel_labels) != REL_LABELS:
        raise RuntimeError("Checkpoint relation-label schema does not match the V14 application schema.")

    embedded_fp = payload.get("training_fingerprint") if isinstance(payload, dict) else None
    manifest_fp = manifest.get("training_fingerprint")
    if embedded_fp and manifest_fp and embedded_fp != manifest_fp:
        raise RuntimeError("Checkpoint training fingerprint differs from student_model_manifest.json.")

    model_id = _model_id_from_sources(manifest, payload)
    resolved_revision = _resolved_revision(manifest, payload)

    from transformers import AutoConfig, AutoTokenizer

    if encoder_config_dir.exists():
        encoder_config = AutoConfig.from_pretrained(
            encoder_config_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
    else:
        config_kwargs = {"trust_remote_code": False}
        if resolved_revision:
            config_kwargs["revision"] = resolved_revision
        encoder_config = AutoConfig.from_pretrained(model_id, **config_kwargs)

    model = DocumentStatementStudent(
        encoder_config=encoder_config,
        n_token_labels=len(TOKEN_LABELS),
        n_rel_labels=len(REL_LABELS),
    )
    model.load_state_dict(_state_dict_from_payload(payload), strict=True)

    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
    else:
        tok_kwargs = {"use_fast": True, "trust_remote_code": False}
        if resolved_revision:
            tok_kwargs["revision"] = resolved_revision
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)

    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for return_offsets_mapping.")

    if device is None or device == "auto":
        resolved_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{resolved_device} was selected, but CUDA is unavailable.")
    if resolved_device.type == "cuda":
        if resolved_device.index is not None and resolved_device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"{resolved_device} does not exist. Visible CUDA devices: {torch.cuda.device_count()}."
            )

    model.to(resolved_device).eval()

    settings = _settings_from_manifest(manifest, relation_threshold)
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        manifest=manifest,
        checkpoint_sha256=checkpoint_sha,
        model_id=model_id,
        resolved_revision=resolved_revision,
        artifact_dir=artifact_dir,
        settings=settings,
        device=resolved_device,
    )


def _decode_bio_window(offsets, pred_ids, token_probs):
    spans = []
    current = None
    for label_id, (a, b), conf in zip(pred_ids, offsets, token_probs):
        if a == b == 0:
            continue
        label = ID2TOKEN[int(label_id)]
        if label == "O":
            if current:
                spans.append(current)
                current = None
            continue

        prefix, typ = label.split("-", 1)
        if prefix == "B" or current is None or current["type"] != typ:
            if current:
                spans.append(current)
            current = {
                "type": typ,
                "start": int(a),
                "end": int(b),
                "scores": [float(conf)],
            }
        else:
            current["end"] = int(b)
            current["scores"].append(float(conf))

    if current:
        spans.append(current)

    for span in spans:
        span["confidence"] = float(np.mean(span.pop("scores")))
    return spans


def _dedupe_predicted_spans(spans: List[dict]) -> List[dict]:
    best: Dict[Tuple[str, int, int], dict] = {}
    for span in spans:
        key = (span["type"], span["start"], span["end"])
        if key not in best or span["confidence"] > best[key]["confidence"]:
            best[key] = span
    return sorted(best.values(), key=lambda x: (x["start"], x["end"], x["type"]))


def _relation_probability(
    model: DocumentStatementStudent,
    hidden: torch.Tensor,
    statement_range: Tuple[int, int],
    argument_range: Tuple[int, int],
    rel_id: int,
) -> float:
    svec = model.span_pool(hidden, statement_range[0], statement_range[1])
    avec = model.span_pool(hidden, argument_range[0], argument_range[1])
    feat = torch.cat(
        [svec, avec, torch.abs(svec - avec), svec * avec],
        dim=-1,
    )
    logits = model.rel_head(feat)
    probs = torch.softmax(logits, dim=-1)
    return float(probs[rel_id].detach().cpu())


def predict_statement_events(
    text: str,
    bundle: ModelBundle,
    doc_id: str = "inference_document",
    relation_threshold: Optional[float] = None,
) -> Tuple[List[dict], List[dict]]:
    """Run the V14 student model and assemble event-centric predictions.

    Deployment-time inference uses only the trained student model. The LLM and
    weak-supervision components are training-time procedures and are never called here.
    """
    if not isinstance(text, str) or not text.strip():
        return [], []

    model = bundle.model
    tokenizer = bundle.tokenizer
    settings = bundle.settings
    threshold = settings.relation_threshold if relation_threshold is None else float(relation_threshold)
    device = bundle.device

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

    all_spans: List[dict] = []
    window_cache: List[dict] = []

    model.eval()
    with torch.inference_mode():
        for wi in range(len(enc["input_ids"])):
            ids = torch.tensor([enc["input_ids"][wi]], dtype=torch.long, device=device)
            mask = torch.tensor([enc["attention_mask"][wi]], dtype=torch.long, device=device)
            offsets = [tuple(x) for x in enc["offset_mapping"][wi]]

            # One encoder pass per sliding window. Relation scoring below reuses
            # these hidden states rather than re-encoding the article for every
            # candidate argument pair.
            encoded = model.encoder(input_ids=ids, attention_mask=mask)
            hidden_batch = model.dropout(encoded.last_hidden_state)
            token_logits = model.token_head(hidden_batch)[0]
            probs = torch.softmax(token_logits, dim=-1)
            pred = probs.argmax(-1)
            conf = probs.max(-1).values

            all_spans.extend(
                _decode_bio_window(
                    offsets,
                    pred.detach().cpu().tolist(),
                    conf.detach().cpu().tolist(),
                )
            )
            window_cache.append(
                {
                    "offsets": offsets,
                    "hidden": hidden_batch[0],
                }
            )

    spans = _dedupe_predicted_spans(all_spans)
    for span in spans:
        span["text"] = text[span["start"] : span["end"]]

    statements = [
        s for s in spans
        if s["type"] in {"STATEMENT_DIRECT", "STATEMENT_INDIRECT"}
    ]
    args = [
        s for s in spans
        if s["type"] in {
            "PERSON",
            "PERSONCOREF",
            "ROLE",
            "AFFILIATION",
            "DATETIME",
            "LOCATION",
            "EVENT",
            "ISSUE",
        }
    ]
    cues = [s for s in spans if s["type"] in {"CUE", "CUECOREF"}]

    expected_rel = {
        "PERSON": "ATTRIBUTED_TO",
        "PERSONCOREF": "ATTRIBUTED_TO",
        "ROLE": "HAS_ROLE",
        "AFFILIATION": "AFFILIATED_WITH",
        "DATETIME": "AT_TIME",
        "LOCATION": "AT_LOCATION",
        "EVENT": "AT_EVENT",
        "ISSUE": "ABOUT_ISSUE",
    }
    field_for_type = {
        "PERSON": "speaker",
        "PERSONCOREF": "speaker",
        "ROLE": "role",
        "AFFILIATION": "affiliation",
        "DATETIME": "datetime",
        "LOCATION": "location",
        "EVENT": "utterance_event",
        "ISSUE": "issue",
    }

    explicit_people = [s for s in spans if s["type"] == "PERSON"]

    def predicted_coref_canonical(arg: dict) -> Optional[str]:
        mention = (arg.get("text") or "").strip()
        mention_norm = mention.lower()
        prior = [p for p in explicit_people if p["end"] <= arg["start"]]

        token_matches = [
            p for p in prior
            if mention_norm
            and mention_norm not in PRONOUNS
            and mention_norm
            in {
                token.lower()
                for token in re.findall(r"[A-Za-zÀ-ÿ.'-]+", p.get("text", ""))
            }
        ]
        if token_matches:
            return token_matches[-1].get("text")
        if prior:
            return prior[-1].get("text")
        return mention or None

    events: List[dict] = []

    for statement_index, statement in enumerate(statements):
        best_fields: Dict[str, dict] = {}

        near_cues = sorted(
            cues,
            key=lambda cue: distance_between(
                statement["start"],
                statement["end"],
                cue["start"],
                cue["end"],
            ),
        )
        cue = None
        if near_cues:
            nearest = near_cues[0]
            if distance_between(
                statement["start"],
                statement["end"],
                nearest["start"],
                nearest["end"],
            ) <= settings.max_cue_char_distance:
                cue = nearest

        for arg in args:
            if distance_between(
                statement["start"],
                statement["end"],
                arg["start"],
                arg["end"],
            ) > settings.max_relation_char_distance:
                continue

            rel_name = expected_rel[arg["type"]]
            rel_id = REL2ID[rel_name]
            best_score = -1.0

            for window in window_cache:
                statement_range = token_range_for_char_span(
                    window["offsets"],
                    statement["start"],
                    statement["end"],
                )
                arg_range = token_range_for_char_span(
                    window["offsets"],
                    arg["start"],
                    arg["end"],
                )
                if statement_range is None or arg_range is None:
                    continue
                with torch.inference_mode():
                    score = _relation_probability(
                        model,
                        window["hidden"],
                        statement_range,
                        arg_range,
                        rel_id,
                    )
                best_score = max(best_score, score)

            if best_score < threshold:
                continue

            field = field_for_type[arg["type"]]
            previous = best_fields.get(field)
            if previous is not None and best_score <= previous["relation_confidence"]:
                continue

            obj = {
                "text": arg["text"],
                "evidence": arg["text"],
                "start": arg["start"],
                "end": arg["end"],
                "label": arg["type"],
                "relation_confidence": best_score,
            }

            if field == "speaker":
                if arg["type"] == "PERSON":
                    obj["canonical"] = arg["text"]
                    obj["mention_label"] = "PERSON"
                    obj["coref_resolution_provenance"] = "EXPLICIT_PREDICTED_PERSON"
                else:
                    obj["canonical"] = predicted_coref_canonical(arg)
                    obj["mention_label"] = "PERSONCOREF"
                    obj["coref_resolution_provenance"] = (
                        "DOCUMENT_HEURISTIC_FROM_PREDICTED_PERSON"
                        if obj.get("canonical") and obj.get("canonical") != arg["text"]
                        else "UNRESOLVED_PERSONCOREF"
                    )

            best_fields[field] = obj

        if cue is not None and cue.get("type") == "CUECOREF" and best_fields.get("speaker"):
            best_fields["speaker"]["source_mention_label"] = "CUECOREF"
            best_fields["speaker"]["coreference_via"] = "CUECOREF"

        event = {
            "event_id": stable_id(
                doc_id,
                statement["start"],
                statement["end"],
                statement_index,
                "prediction",
                prefix="stmt",
            ),
            "doc_id": doc_id,
            "statement_type": (
                "DIRECT"
                if statement["type"] == "STATEMENT_DIRECT"
                else "INDIRECT"
            ),
            "statement": {
                "text": statement["text"],
                "evidence": statement["text"],
                "start": statement["start"],
                "end": statement["end"],
                "confidence": statement["confidence"],
            },
            "cue": (
                None
                if cue is None
                else {
                    "text": cue["text"],
                    "evidence": cue["text"],
                    "start": cue["start"],
                    "end": cue["end"],
                    "confidence": cue["confidence"],
                    "label": cue["type"],
                }
            ),
            "speaker": best_fields.get("speaker"),
            "role": best_fields.get("role"),
            "affiliation": best_fields.get("affiliation"),
            "datetime": best_fields.get("datetime"),
            "location": best_fields.get("location"),
            "utterance_event": best_fields.get("utterance_event"),
            "issue": best_fields.get("issue"),
            "span_provenance": "STUDENT_MODEL_PREDICTION",
            "relation_provenance": "STUDENT_MODEL_PREDICTION",
        }
        events.append(event)

    return events, spans


def _cache_fingerprint(
    text: str,
    doc_id: str,
    bundle: ModelBundle,
    relation_threshold: float,
) -> str:
    payload = {
        "schema_version": APP_INFERENCE_SCHEMA_VERSION,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "training_fingerprint": bundle.manifest.get("training_fingerprint"),
        "doc_id": doc_id,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "max_length": bundle.settings.max_length,
        "stride": bundle.settings.stride,
        "relation_threshold": float(relation_threshold),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]


def infer_with_disk_cache(
    text: str,
    doc_id: str,
    bundle: ModelBundle,
    relation_threshold: float,
    cache_dir: Path,
    use_cache: bool = True,
) -> Tuple[List[dict], List[dict], bool]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _cache_fingerprint(text, doc_id, bundle, relation_threshold)
    cache_file = cache_dir / f"{doc_id}_{fingerprint}.json"

    if use_cache and cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                payload.get("inference_schema_version") == APP_INFERENCE_SCHEMA_VERSION
                and payload.get("fingerprint") == fingerprint
            ):
                return payload.get("events", []), payload.get("spans", []), True
        except Exception:
            pass

    events, spans = predict_statement_events(
        text,
        bundle,
        doc_id=doc_id,
        relation_threshold=relation_threshold,
    )

    payload = {
        "inference_schema_version": APP_INFERENCE_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "doc_id": doc_id,
        "events": events,
        "spans": spans,
    }

    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(cache_file)

    return events, spans, False


def slug(text: str) -> str:
    value = re.sub(r"[^\w]+", "_", text.strip().lower(), flags=re.UNICODE).strip("_")
    return value[:120] or "unknown"


def _add_entity_node(
    graph: nx.MultiDiGraph,
    prefix: str,
    obj: Optional[dict],
    node_type: str,
) -> Optional[str]:
    if not obj:
        return None
    label = obj.get("canonical") or obj.get("text") or obj.get("evidence")
    if not label:
        return None
    node_id = f"{prefix}:{slug(str(label))}"
    graph.add_node(node_id, node_type=node_type, label=str(label))
    return node_id


def build_nx_kg(events: List[dict]) -> nx.MultiDiGraph:
    """Build the formal event-centric model-generated KG used by V14.

    Cue is retained as evidence/metadata on the StatementEvent node rather than
    introduced as a new semantic relation.
    """
    graph = nx.MultiDiGraph()

    for event in events:
        doc_node = f"article:{event['doc_id']}"
        graph.add_node(
            doc_node,
            node_type="Article",
            label=event["doc_id"],
        )

        statement = event.get("statement") or {}
        event_node = f"statement_event:{event['event_id']}"
        cue = event.get("cue") or {}
        graph.add_node(
            event_node,
            node_type="StatementEvent",
            label=statement.get("text", ""),
            statement_type=event.get("statement_type"),
            content=statement.get("text", ""),
            evidence_start=int(statement.get("start", -1)),
            evidence_end=int(statement.get("end", -1)),
            confidence=float(statement.get("confidence", 0.0)),
            cue_text=cue.get("text", ""),
            cue_label=cue.get("label", ""),
            cue_confidence=float(cue.get("confidence", 0.0)) if cue else 0.0,
            span_provenance=event.get("span_provenance", ""),
            relation_provenance=event.get("relation_provenance", ""),
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

            edge_attrs = {"relation": relation}
            if obj.get("start") is not None:
                edge_attrs.update(
                    {
                        "evidence_start": int(obj["start"]),
                        "evidence_end": int(obj["end"]),
                        "evidence_text": obj.get("evidence", obj.get("text", "")),
                    }
                )
            if obj.get("relation_confidence") is not None:
                edge_attrs["relation_confidence"] = float(obj["relation_confidence"])
            if obj.get("label") is not None:
                edge_attrs["source_span_label"] = str(obj["label"])
            if obj.get("mention_label") is not None:
                edge_attrs["source_mention_label"] = str(obj["mention_label"])
            if obj.get("source_mention_label") is not None:
                edge_attrs["source_mention_label"] = str(obj["source_mention_label"])
            if obj.get("coreference_via") is not None:
                edge_attrs["coreference_via"] = str(obj["coreference_via"])
            if obj.get("coref_resolution_provenance") is not None:
                edge_attrs["coref_resolution_provenance"] = str(
                    obj["coref_resolution_provenance"]
                )
            graph.add_edge(event_node, target, **edge_attrs)

    return graph


PFSA = Namespace("https://example.org/pfsa/")


def _uri(kind: str, value: str) -> URIRef:
    return URIRef(str(PFSA) + kind + "/" + slug(value))


def build_rdf_kg(events: List[dict]) -> Graph:
    graph = Graph()
    graph.bind("pfsa", PFSA)

    for event in events:
        ev = _uri("statement-event", event["event_id"])
        article = _uri("article", event["doc_id"])
        statement = event.get("statement") or {}
        cue = event.get("cue") or {}

        graph.add((ev, RDF.type, PFSA.StatementEvent))
        graph.add((article, RDF.type, PFSA.Article))
        graph.add((ev, PFSA.sourceArticle, article))
        graph.add((ev, PFSA.statementType, Literal(event.get("statement_type", ""))))
        graph.add((ev, PFSA.content, Literal(statement.get("text", ""), lang="id")))
        graph.add(
            (
                ev,
                PFSA.evidenceStart,
                Literal(int(statement.get("start", -1)), datatype=XSD.integer),
            )
        )
        graph.add(
            (
                ev,
                PFSA.evidenceEnd,
                Literal(int(statement.get("end", -1)), datatype=XSD.integer),
            )
        )
        graph.add(
            (
                ev,
                PFSA.confidence,
                Literal(float(statement.get("confidence", 0.0)), datatype=XSD.double),
            )
        )
        graph.add((ev, PFSA.spanProvenance, Literal(event.get("span_provenance", ""))))
        graph.add((ev, PFSA.relationProvenance, Literal(event.get("relation_provenance", ""))))

        if cue:
            graph.add((ev, PFSA.cueText, Literal(cue.get("text", ""), lang="id")))
            graph.add((ev, PFSA.cueLabel, Literal(cue.get("label", ""))))
            graph.add(
                (
                    ev,
                    PFSA.cueConfidence,
                    Literal(float(cue.get("confidence", 0.0)), datatype=XSD.double),
                )
            )

        mapping = [
            ("speaker", "person", PFSA.Person, PFSA.speaker),
            ("role", "role", PFSA.Role, PFSA.hasRole),
            ("affiliation", "organization", PFSA.Organization, PFSA.affiliatedWith),
            ("datetime", "datetime", PFSA.DateTime, PFSA.atTime),
            ("location", "location", PFSA.Location, PFSA.atLocation),
            ("utterance_event", "utterance-event", PFSA.UtteranceEvent, PFSA.atEvent),
            ("issue", "issue", PFSA.Issue, PFSA.aboutIssue),
        ]

        for field, kind, rdf_type, predicate in mapping:
            obj = event.get(field)
            if not obj:
                continue
            label = obj.get("canonical") or obj.get("text") or obj.get("evidence")
            if not label:
                continue
            target = _uri(kind, str(label))
            graph.add((target, RDF.type, rdf_type))
            graph.add((target, PFSA.label, Literal(str(label), lang="id")))
            graph.add((ev, predicate, target))

    return graph


def _event_value(event: dict, field: str) -> Optional[str]:
    obj = event.get(field)
    if not obj:
        return None
    return obj.get("canonical") or obj.get("text") or obj.get("evidence")


def events_to_dataframe(events: List[dict]) -> pd.DataFrame:
    rows = []
    for event in events:
        statement = event.get("statement") or {}
        cue = event.get("cue") or {}
        speaker = event.get("speaker") or {}
        rows.append(
            {
                "doc_id": event.get("doc_id"),
                "event_id": event.get("event_id"),
                "statement_type": event.get("statement_type"),
                "statement": statement.get("text"),
                "statement_confidence": statement.get("confidence"),
                "cue": cue.get("text"),
                "cue_label": cue.get("label"),
                "cue_confidence": cue.get("confidence"),
                "speaker": _event_value(event, "speaker"),
                "speaker_mention": speaker.get("text"),
                "speaker_mention_label": speaker.get("mention_label"),
                "coreference_via": speaker.get("coreference_via"),
                "speaker_relation_confidence": speaker.get("relation_confidence"),
                "role": _event_value(event, "role"),
                "role_relation_confidence": (event.get("role") or {}).get("relation_confidence"),
                "affiliation": _event_value(event, "affiliation"),
                "affiliation_relation_confidence": (event.get("affiliation") or {}).get("relation_confidence"),
                "datetime": _event_value(event, "datetime"),
                "location": _event_value(event, "location"),
                "utterance_event": _event_value(event, "utterance_event"),
                "issue": _event_value(event, "issue"),
                "evidence_start": statement.get("start"),
                "evidence_end": statement.get("end"),
                "span_provenance": event.get("span_provenance"),
                "relation_provenance": event.get("relation_provenance"),
            }
        )
    return pd.DataFrame(rows)


def spans_to_dataframe(spans: List[dict], doc_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "doc_id": doc_id,
                "type": span.get("type"),
                "text": span.get("text"),
                "start": span.get("start"),
                "end": span.get("end"),
                "confidence": span.get("confidence"),
            }
            for span in spans
        ]
    )


def relation_rows_for_event(event: dict) -> List[dict]:
    mapping = [
        ("speaker", "ATTRIBUTED_TO"),
        ("role", "HAS_ROLE"),
        ("affiliation", "AFFILIATED_WITH"),
        ("datetime", "AT_TIME"),
        ("location", "AT_LOCATION"),
        ("utterance_event", "AT_EVENT"),
        ("issue", "ABOUT_ISSUE"),
    ]
    rows = []
    for field, relation in mapping:
        obj = event.get(field)
        if not obj:
            continue
        rows.append(
            {
                "relation": relation,
                "field": field,
                "value": obj.get("canonical") or obj.get("text") or obj.get("evidence"),
                "evidence": obj.get("evidence"),
                "span_label": obj.get("label"),
                "mention_label": obj.get("mention_label"),
                "coreference_via": obj.get("coreference_via"),
                "confidence": obj.get("relation_confidence"),
                "start": obj.get("start"),
                "end": obj.get("end"),
            }
        )
    return rows


def events_to_jsonl_bytes(events: List[dict]) -> bytes:
    suffix = "\n" if events else ""
    return (
        "\n".join(json.dumps(row, ensure_ascii=False) for row in events) + suffix
    ).encode("utf-8")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def graphml_bytes(graph: nx.MultiDiGraph) -> bytes:
    return ("\n".join(nx.generate_graphml(graph)) + "\n").encode("utf-8")


def rdf_bytes(events: List[dict], fmt: str) -> bytes:
    graph = build_rdf_kg(events)
    serialized = graph.serialize(format=fmt)
    return serialized if isinstance(serialized, bytes) else serialized.encode("utf-8")


def graph_edges_dataframe(graph: nx.MultiDiGraph) -> pd.DataFrame:
    rows = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        rows.append(
            {
                "source": source,
                "source_label": graph.nodes[source].get("label"),
                "source_type": graph.nodes[source].get("node_type"),
                "target": target,
                "target_label": graph.nodes[target].get("label"),
                "target_type": graph.nodes[target].get("node_type"),
                "edge_key": key,
                "relation": attrs.get("relation"),
                "relation_confidence": attrs.get("relation_confidence"),
                "source_span_label": attrs.get("source_span_label"),
                "source_mention_label": attrs.get("source_mention_label"),
                "coreference_via": attrs.get("coreference_via"),
                "coref_resolution_provenance": attrs.get("coref_resolution_provenance"),
                "evidence_text": attrs.get("evidence_text"),
                "evidence_start": attrs.get("evidence_start"),
                "evidence_end": attrs.get("evidence_end"),
            }
        )
    return pd.DataFrame(rows)
