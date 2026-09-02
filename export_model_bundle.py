from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

STATEMENT_ENTITY_TYPES = ["STATEMENT_DIRECT", "STATEMENT_INDIRECT"]
ENTITY_CUE_TYPES = ["PERSON", "PERSONCOREF", "ROLE", "AFFILIATION", "DATETIME", "LOCATION", "EVENT", "CUE", "CUECOREF"]
STATEMENT_LABELS = ["O"] + [f"{p}-{t}" for t in STATEMENT_ENTITY_TYPES for p in ("B", "I", "L", "U")]
ENTITY_LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_CUE_TYPES for p in ("B", "I", "L", "U")]
REL_LABELS = ["NONE", "ATTRIBUTED_TO", "HAS_ROLE", "AFFILIATED_WITH", "AT_TIME", "AT_LOCATION", "AT_EVENT"]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def copytree_replace(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def find_teacher_model(source: Path):
    candidates = [source / "experiment_lock.json", source / "experiment_manifest_final.json", source / "llm_annotation_audit.json"]
    for path in candidates:
        if not path.exists():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        text = json.dumps(obj, ensure_ascii=False)
        for model in ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B"]:
            if model in text:
                return model
    return None


def main():
    parser = argparse.ArgumentParser(description="Export the clean PFSA-ID student model for Streamlit deployment.")
    parser.add_argument("--source", required=True, type=Path, help="Completed experiment directory")
    parser.add_argument("--destination", default=Path("model_artifacts"), type=Path)
    args = parser.parse_args()

    source, destination = args.source.resolve(), args.destination.resolve()
    checkpoint = source / "student_model_best.pt"
    manifest_path = source / "student_model_manifest.json"
    tokenizer_dir = source / "student_tokenizer"

    missing = [str(p) for p in [checkpoint, manifest_path, tokenizer_dir] if not p.exists()]
    if missing:
        raise SystemExit("Missing required artifacts:\n- " + "\n- ".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_complete") is not True:
        raise SystemExit("Refusing to export: training_complete is not true.")

    checkpoint_sha = sha256_file(checkpoint)
    expected_sha = (manifest.get("checkpoint") or {}).get("sha256")
    if expected_sha and expected_sha != checkpoint_sha:
        raise SystemExit("Refusing to export: checkpoint SHA-256 does not match the training manifest.")

    payload = safe_torch_load(checkpoint)
    if list(payload.get("statement_labels", [])) != STATEMENT_LABELS:
        raise SystemExit("Checkpoint does not use the expected Statement BILUO schema.")
    if list(payload.get("entity_labels", [])) != ENTITY_LABELS:
        raise SystemExit("Checkpoint does not use the expected Entity/Cue BILUO schema.")
    if list(payload.get("relation_labels", [])) != REL_LABELS:
        raise SystemExit("Checkpoint relation schema is incompatible.")

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination / checkpoint.name)
    shutil.copy2(manifest_path, destination / manifest_path.name)
    copytree_replace(tokenizer_dir, destination / "student_tokenizer")

    training_config = manifest.get("fingerprint_payload", {}).get("training_config", {})
    model_id = payload.get("student_model_id") or training_config.get("student_model_id", "indobenchmark/indobert-base-p1")
    resolved_revision = payload.get("student_resolved_revision") or training_config.get("student_resolved_revision")

    encoder_config_src = source / "student_encoder_config"
    if encoder_config_src.exists():
        copytree_replace(encoder_config_src, destination / "student_encoder_config")
    else:
        from transformers import AutoConfig
        kwargs = {"trust_remote_code": False}
        if resolved_revision:
            kwargs["revision"] = str(resolved_revision)
        config = AutoConfig.from_pretrained(str(model_id), **kwargs)
        config.save_pretrained(destination / "student_encoder_config")

    optional = [
        "student_training_current_manifest.json", "experiment_lock.json", "model_revision_lock.json",
        "experiment_manifest_final.json", "requirements_frozen.txt", "article_split.json",
        "llm_annotation_audit.json", "llm_quality_summary.json", "student_test_metrics.json",
    ]
    for name in optional:
        p = source / name
        if p.exists():
            shutil.copy2(p, destination / name)

    bundle_manifest = {
        "bundle_schema_version": 3,
        "deployment_architecture": "dual_biluo_crf_issue_sentence_relation",
        "training_fingerprint": manifest.get("training_fingerprint"),
        "checkpoint_sha256": checkpoint_sha,
        "student_model_id": model_id,
        "student_resolved_revision": resolved_revision,
        "teacher_model_id": find_teacher_model(source),
        "statement_labels": STATEMENT_LABELS,
        "entity_labels": ENTITY_LABELS,
        "relation_labels": REL_LABELS,
        "issue_schema": "COMPLETE_SENTENCE_BINARY_HEAD",
        "inference_speaker_max_distance": 500,
        "inference_profile_max_distance": 260,
        "inference_context_arg_max_distance": 500,
        "inference_cue_max_distance": 160,
    }
    (destination / "deployment_bundle_manifest.json").write_text(json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("Deployment bundle exported to:", destination)
    print("Checkpoint SHA-256:", checkpoint_sha)
    print("Student model:", model_id)
    print("Teacher model:", bundle_manifest["teacher_model_id"] or "not found in experiment metadata")


if __name__ == "__main__":
    main()
