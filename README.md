# PFSA-ID Statement–Event Explorer

A streamlined Streamlit demo for the deployment stage of the clean PFSA-ID pipeline.

```text
Indonesian news article
        ↓
trained document-level student model
        ↓
Statement BILUO-CRF + Entity/Cue BILUO-CRF
        ↓
ISSUE sentence-level head + relation head
        ↓
StatementEvent assembly
        ↓
evidence-grounded knowledge graph
```

The application does **not** run Qwen during deployment. Qwen2.5-3B-Instruct can remain the teacher used during training/silver annotation; the live app loads only the final student-model checkpoint.

## Export the model artifact

After the clean notebook finishes training:

```bash
python export_model_bundle.py \
  --source /path/to/pfsa_experiments/<experiment> \
  --destination ./model_artifacts
```

The exporter checks:

- `training_complete=True`;
- checkpoint SHA-256;
- Statement BILUO schema;
- Entity/Cue BILUO schema;
- relation schema;
- local tokenizer and encoder configuration.

## Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

For a shared GPU server, isolate one physical GPU before starting Streamlit:

```bash
CUDA_VISIBLE_DEVICES=1 streamlit run app.py
```

Inside the app that GPU appears as logical `cuda:0`.

## User flow

1. Load the model from the sidebar.
2. Choose an example or paste an Indonesian news article.
3. Click **Analyze article**.
4. Inspect extracted StatementEvents, evidence spans, and the interactive knowledge graph.

Technical controls are placed under **Advanced settings** to keep the main interface simple.

## Knowledge-graph relations

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

`CUE` and `CUECOREF` remain extraction evidence rather than separate KG relations.
