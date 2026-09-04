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
- a **Previous / Next** navigator for sequential statement inspection;
- a prominent three-button view menu: **Extraction / Evidence / Knowledge Graph**;
- the active view is preserved while browsing statements;
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
6. Browse extracted statements with **Previous / Next** or the statement selector.
7. Switch between the clearly separated **Extraction / Evidence / Knowledge Graph** views.

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


## Evidence view by statement type

The Evidence view follows the global **Statement type** filter:

- **All**: highlights every DIRECT and INDIRECT statement in the article and the ISSUE evidence related to those events.
- **Direct**: highlights only DIRECT statements and their related ISSUE evidence.
- **Indirect**: highlights only INDIRECT statements and their related ISSUE evidence.

Evidence is therefore article-level. The statement navigator continues to control the currently selected event for Extraction and selected-statement graph inspection.

## Layered Evidence rendering

The Evidence view preserves the full statement-event annotation layer.

- **All** shows evidence from every DIRECT and INDIRECT StatementEvent.
- **Direct** shows all evidence fields belonging to DIRECT StatementEvents only.
- **Indirect** shows all evidence fields belonging to INDIRECT StatementEvents only.

For every visible event, the renderer keeps `STATEMENT`, `CUE/CUECOREF`,
`PERSON/PERSONCOREF`, `ROLE`, `AFFILIATION`, `DATETIME`, `LOCATION`, `EVENT`,
and `ISSUE` where those fields have grounded character offsets. When a specific
label overlaps a STATEMENT span, the specific label keeps its own color and a
thin outline indicates that the text also belongs to the statement span.

## Article input

The app automatically reads `sample_news.csv` from the same directory as `app.py`. Each valid row must contain at least `doc_id` and `text`; optional `title`, `name`, or `label` columns are used for friendlier dropdown labels.

Users can choose **Sample news** to load a sample article automatically or **Paste text** to enter their own article. Switching input sources does not modify the CSV, and stale extraction results are hidden until the newly selected article is analyzed.

## Sample CSV encoding

`sample_news.csv` is loaded automatically using the first compatible encoding
from `utf-8-sig`, `utf-8`, `cp1252`, and `latin1`. The UI shows the encoding
actually used. If the CSV cannot be parsed, the app explicitly reports the error
before falling back to built-in demonstration samples.
