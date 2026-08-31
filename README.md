# PFSA-ID V14 Statement–Event Explorer

A Streamlit demo for the **deployment-time** path of the PFSA-ID V14 research pipeline:

```text
Indonesian news article
    ↓
trained document-level student model
    ↓
BIO span extraction
    ↓
statement–argument relation prediction
    ↓
StatementEvent assembly
    ↓
evidence-grounded knowledge graph
```

The application does **not** run the LLM annotator or weak supervision. Those are training-time components. The deployed application uses only the verified trained student model.

## What is improved in this version

- Aligned with the V14 token schema:
  - `STATEMENT_DIRECT`
  - `STATEMENT_INDIRECT`
  - `PERSON`
  - `PERSONCOREF`
  - `ROLE`
  - `AFFILIATION`
  - `DATETIME`
  - `LOCATION`
  - `EVENT`
  - `ISSUE`
  - `CUE`
  - `CUECOREF`
- Aligned with the V14 relation schema, including `ABOUT_ISSUE`.
- Preserves source-coreference metadata (`PERSONCOREF`, `CUECOREF`, canonical speaker, resolution provenance).
- One encoder pass per sliding window; relation scoring reuses the same hidden states, making the live demo much faster than repeatedly re-encoding the article for every relation pair.
- Character-level evidence view for a selected StatementEvent.
- Interactive event-centric knowledge graph.
- Relation filtering and statement-event focus.
- Model artifact verification using checkpoint SHA-256 and training fingerprint.
- Explicit CPU / visible-GPU selection from the Streamlit sidebar.
- Model load/release controls to avoid unintentionally keeping stale GPU allocations.
- Disk inference cache fingerprinted by model, article, threshold, and inference schema.
- GraphML, Turtle, JSON-LD, event JSONL/CSV, and edge-list CSV exports.
- Synthetic built-in news examples for demonstrations.

## 1. Export the trained V14 model

After the V14 notebook has completed training, its experiment directory should contain at least:

```text
<experiment>/
├── student_model_best.pt
├── student_model_manifest.json
└── student_tokenizer/
```

Recommended reproducibility files include:

```text
experiment_lock.json
model_revision_lock.json
experiment_manifest_final.json
requirements_frozen.txt
article_split.json
```

Export a standalone deployment bundle:

```bash
python export_model_bundle.py \
  --source /path/to/pfsa_experiments/V14_FINAL_001 \
  --destination ./model_artifacts
```

The exporter verifies:

- `training_complete=True`
- checkpoint SHA-256
- V14 token labels
- V14 relation labels
- pinned student-model revision where available

It also saves `student_encoder_config/`, so the app can reconstruct the model architecture without downloading pretrained weights again.

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Run

```bash
streamlit run app.py
```

The default artifact directory is:

```text
./model_artifacts
```

You can change it in the sidebar or set:

```bash
export PFSA_MODEL_ARTIFACT_DIR=/path/to/model_artifacts
```

Optional default device:

```bash
export PFSA_APP_DEVICE=cuda:0
```

### GPU selection

The sidebar lists the CUDA devices visible to the Streamlit process:

```text
auto
cpu
cuda:0
cuda:1
...
```

If the server uses `CUDA_VISIBLE_DEVICES`, these are **logical visible device IDs**. For example:

```bash
CUDA_VISIBLE_DEVICES=2 streamlit run app.py
```

will expose physical GPU 2 to the app as logical `cuda:0`.

For a shared multi-GPU server, this launch pattern is preferable because it isolates the application to one physical GPU.

## 4. Live demo

The **Live Demo · One Article** tab contains synthetic demonstration articles. They are clearly marked as synthetic and are not factual news claims.

Click:

```text
Run inference → build knowledge graph
```

The application then shows:

1. extracted StatementEvents,
2. direct vs indirect counts,
3. source/cue/content and contextual arguments,
4. exact evidence spans in the original article,
5. interactive event-centric KG,
6. downloadable machine-readable artifacts,
7. checkpoint, training fingerprint, model revision, device, and inference provenance.

## 5. KG semantics

Formal relations:

```text
StatementEvent ──ATTRIBUTED_TO────▶ Person
StatementEvent ──HAS_ROLE─────────▶ Role
StatementEvent ──AFFILIATED_WITH──▶ Organization
StatementEvent ──AT_TIME──────────▶ DateTime
StatementEvent ──AT_LOCATION──────▶ Location
StatementEvent ──AT_EVENT─────────▶ UtteranceEvent
StatementEvent ──ABOUT_ISSUE──────▶ Issue
StatementEvent ──SOURCE_ARTICLE───▶ Article
```

`CUE` / `CUECOREF` remains evidence metadata on the StatementEvent instead of being promoted to an additional formal KG relation. `PERSONCOREF` remains distinct at the extraction layer; when a conservative antecedent can be resolved from an earlier predicted `PERSON`, the canonical person and resolution provenance are retained.

## 6. Batch demo

Upload CSV / Excel, select:

- article text column,
- optional document ID column,

then run batch inference. The app can visualize one document or the aggregate model-generated graph.

A sample file is included:

```text
sample_news.csv
```

## 7. Important methodological note

The graph displayed by this app is a **model-generated KG**, not the gold/silver reference training KG.

Therefore the demo corresponds to the research claim:

```text
load trained student model
→ infer on unseen news
→ extract evidence-grounded statement events
→ construct model-generated KG
```

No graph element is added merely for visual completeness when the model does not predict the supporting span/relation.
