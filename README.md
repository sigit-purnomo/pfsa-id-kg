# PFSA-ID Statement-Event Explorer

A streamlined Streamlit deployment app for the balanced PFSA-ID student model.

```text
Indonesian news article
        |
        v
trained document-level student model
        |
        +--> Statement BILUO-CRF
        |      +--> STATEMENT_DIRECT
        |      +--> STATEMENT_INDIRECT
        |
        +--> Entity/Cue BILUO-CRF
        +--> ISSUE sentence-level head
        +--> Relation head
        |
        v
DIRECT + INDIRECT StatementEvents
        |
        v
evidence-grounded knowledge graph
```

The deployment app does **not** run Qwen. Qwen2.5-3B-Instruct may remain the training-time teacher used to produce silver indirect supervision; live inference loads only the final IndoBERT-based student checkpoint.

## What the UI shows

- total predicted statements;
- DIRECT statement count;
- INDIRECT statement count;
- statement-type filter: **All / Direct / Indirect**;
- exact source evidence and BILUO spans;
- statement-event relations;
- an interactive knowledge graph where:
  - blue StatementEvent nodes are **DIRECT**;
  - amber StatementEvent nodes are **INDIRECT**;
  - relation edges inherit the statement type color.

The filter changes only the display. Model inference always decodes both `STATEMENT_DIRECT` and `STATEMENT_INDIRECT` labels in the same pass.

## Export the balanced model artifact

After the clean balanced notebook finishes training:

```bash
python export_model_bundle.py \
  --source /path/to/pfsa_experiments/<experiment> \
  --destination ./model_artifacts
```

The exporter verifies:

- `training_complete=True`;
- checkpoint SHA-256;
- DIRECT + INDIRECT Statement BILUO labels;
- Entity/Cue BILUO schema;
- relation schema;
- tokenizer and encoder configuration.

When available, the deployment manifest also stores the checkpoint-selection weights used during training, including the explicit DIRECT and INDIRECT contributions.

## Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

For a shared GPU server:

```bash
CUDA_VISIBLE_DEVICES=1 streamlit run app.py
```

The selected physical GPU then appears as logical `cuda:0` inside the application.

## User flow

1. Load the model artifact from the sidebar.
2. Paste an Indonesian news article or use the DIRECT + INDIRECT example.
3. Click **Analyze article**.
4. Review total, DIRECT, and INDIRECT extraction counts.
5. Filter statement type if needed.
6. Inspect extraction evidence and the Knowledge Graph.

## Knowledge graph

```text
StatementEvent [DIRECT or INDIRECT]
    |-- ATTRIBUTED_TO -----> Person
    |-- HAS_ROLE ----------> Role
    |-- AFFILIATED_WITH ---> Organization
    |-- AT_TIME -----------> DateTime
    |-- AT_LOCATION -------> Location
    |-- AT_EVENT ----------> UtteranceEvent
    |-- ABOUT_ISSUE -------> Issue
    `-- SOURCE_ARTICLE ----> Article
```

`CUE` and `CUECOREF` remain extraction evidence rather than standalone KG relations.
