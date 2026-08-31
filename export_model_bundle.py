from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


V14_TOKEN_ENTITY_TYPES = [
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
V14_TOKEN_LABELS = ["O"] + [f"{p}-{t}" for t in V14_TOKEN_ENTITY_TYPES for p in ("B", "I")]
V14_REL_LABELS = [
    "NONE",
    "ATTRIBUTED_TO",
    "HAS_ROLE",
    "AFFILIATED_WITH",
    "AT_TIME",
    "AT_LOCATION",
    "AT_EVENT",
    "ABOUT_ISSUE",
]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def copytree_replace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def safe_torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a completed PFSA-ID V14 student model into a standalone "
            "Streamlit deployment bundle."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "V14 experiment output directory containing student_model_best.pt, "
            "student_model_manifest.json, and student_tokenizer/."
        ),
    )
    parser.add_argument(
        "--destination",
        default="model_artifacts",
        help="Destination deployment bundle directory.",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()

    checkpoint = source / "student_model_best.pt"
    manifest_path = source / "student_model_manifest.json"
    tokenizer_dir = source / "student_tokenizer"

    required = [checkpoint, manifest_path, tokenizer_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required V14 artifacts:\n- " + "\n- ".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_complete") is not True:
        raise SystemExit(
            "Refusing to export: student_model_manifest.json does not mark training_complete=True."
        )

    checkpoint_sha = sha256_file(checkpoint)
    expected_sha = (manifest.get("checkpoint") or {}).get("sha256")
    if expected_sha and expected_sha != checkpoint_sha:
        raise SystemExit(
            "Refusing to export: checkpoint SHA-256 does not match the V14 training manifest."
        )

    payload = safe_torch_load(checkpoint)
    if isinstance(payload, dict):
        token_labels = payload.get("token_labels")
        relation_labels = payload.get("relation_labels")
        if token_labels is not None and list(token_labels) != V14_TOKEN_LABELS:
            raise SystemExit(
                "Refusing to export: checkpoint does not use the V14 token-label schema."
            )
        if relation_labels is not None and list(relation_labels) != V14_REL_LABELS:
            raise SystemExit(
                "Refusing to export: checkpoint does not use the V14 relation-label schema."
            )

    destination.mkdir(parents=True, exist_ok=True)

    shutil.copy2(checkpoint, destination / checkpoint.name)
    shutil.copy2(manifest_path, destination / manifest_path.name)
    copytree_replace(tokenizer_dir, destination / "student_tokenizer")

    for optional_name in [
        "student_training_current_manifest.json",
        "experiment_lock.json",
        "model_revision_lock.json",
        "experiment_manifest_final.json",
        "requirements_frozen.txt",
        "article_split.json",
    ]:
        optional_source = source / optional_name
        if optional_source.exists():
            shutil.copy2(optional_source, destination / optional_name)

    training_config = (
        manifest.get("fingerprint_payload", {})
        .get("training_config", {})
    )
    model_id = (
        payload.get("student_model_id")
        if isinstance(payload, dict) and payload.get("student_model_id")
        else training_config.get("student_model_id", "indobenchmark/indobert-base-p1")
    )
    resolved_revision = (
        payload.get("student_resolved_revision")
        if isinstance(payload, dict) and payload.get("student_resolved_revision")
        else training_config.get("student_resolved_revision")
    )

    existing_encoder_config = source / "student_encoder_config"
    if existing_encoder_config.exists():
        copytree_replace(existing_encoder_config, destination / "student_encoder_config")
        encoder_config_source = "copied_from_experiment"
    else:
        from transformers import AutoConfig

        config_kwargs = {"trust_remote_code": False}
        if resolved_revision:
            config_kwargs["revision"] = str(resolved_revision)
        config = AutoConfig.from_pretrained(str(model_id), **config_kwargs)
        config.save_pretrained(destination / "student_encoder_config")
        encoder_config_source = "resolved_from_pinned_model_revision"

    bundle_manifest = {
        "bundle_version": 2,
        "compatible_pipeline": "PFSA-ID V14",
        "training_fingerprint": manifest.get("training_fingerprint"),
        "training_manifest_schema_version": manifest.get("manifest_schema_version"),
        "checkpoint_sha256": checkpoint_sha,
        "student_model_id": model_id,
        "student_resolved_revision": resolved_revision,
        "encoder_config_source": encoder_config_source,
        "v14_token_labels": V14_TOKEN_LABELS,
        "v14_relation_labels": V14_REL_LABELS,
        "files": {
            "checkpoint": "student_model_best.pt",
            "training_manifest": "student_model_manifest.json",
            "tokenizer": "student_tokenizer/",
            "encoder_config": "student_encoder_config/",
        },
    }

    (destination / "deployment_bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Deployment bundle exported to: {destination}")
    print(f"Checkpoint SHA-256: {checkpoint_sha}")
    print(f"Student model: {model_id}")
    print(f"Pinned revision: {resolved_revision}")


if __name__ == "__main__":
    main()
