# Clinical GraphRAG Evaluation

Extract structured Knowledge Graphs from clinical transcripts and evaluate them against a human-curated reference. Built for the Chicago AI Science hackathon.

## Background

A doctor has a 15-minute appointment with a patient. That conversation is recorded and transcribed. The transcript is dense and unstructured — symptoms scattered throughout, past history referenced mid-conversation, treatments discussed in passing. If another clinician needs to quickly understand that patient's situation later (a specialist, an ER doctor, a nurse during handoff), reading the full transcript is slow and error-prone.

A Knowledge Graph pulls out the structured facts — *this patient has COPD, reported shortness of breath, is on albuterol, has a history of smoking* — and the relationships between them. That's what a doctor mentally does when reading a chart: building a model of entities and how they connect. The richer and more accurate the KG, the better a downstream GraphRAG system can answer questions about the patient.

**Clinically, this matters because it:**
- Reduces documentation burden — the KG can auto-populate a structured patient summary
- Enables faster care handoffs
- Supports clinical decision support (flagging drug interactions, missing follow-ups)
- Scales across large patient populations for retrospective analysis

## Hackathon Goal

**Build a multi-agent KG extraction pipeline that produces a Knowledge Graph as close as possible to the human-curated reference — at the lowest API cost.**

A naive single-pass baseline is provided in `kg_extraction.py` (composite score: **0.562**). Your task is to design better agentic orchestration that extracts higher-quality KGs from clinical transcripts.

### What is a Knowledge Graph here?

A KG captures **entities** (symptoms, diagnoses, treatments, procedures, etc.) and their **relationships** extracted from patient transcripts.

Node types:

| Type | Examples |
|------|---------|
| `SYMPTOM` | shortness of breath, dry cough, chest tightness |
| `DIAGNOSIS` | COPD exacerbation, covid-19, upper respiratory infection |
| `TREATMENT` | albuterol, tylenol, 14-day isolation |
| `PROCEDURE` | covid swab, chest x-ray, lung auscultation |
| `LOCATION` | chest, throat, lungs |
| `MEDICAL_HISTORY` | smoking, hypertension, type 1 diabetes |
| `LAB_RESULT` | temperature 101 F, BP 148/90, A1C 7.2% |

Edge types: `INDICATES`, `RULES_OUT`, `CAUSES`, `LOCATED_AT`, `TAKEN_FOR`, `CONFIRMS`

This structured representation is used by **GraphRAG** to retrieve relevant clinical context when answering questions about a patient.

### Scoring

#### Development metric: Composite Score

Run `kg_similarity_scorer.py` at any time to measure how close your KG is to the human-curated reference:

| Component | Weight | Description |
|-----------|--------|-------------|
| Entity F1 | 25% | Semantic node overlap with curated KG |
| Population Completeness | 25% | Node count coverage |
| Relation Completeness | 25% | Edge count coverage |
| Schema Completeness | 25% | Node type coverage |

| Method | Composite Score |
|--------|-----------------|
| Naive single-pass (GLM) | 0.562 |

Higher = closer to what a human expert would extract.

#### Final evaluation: LLM Judge (run by organizers)

After submission, organizers run a **GraphRAG QA test**: clinical questions are answered using your KG, and those answers are scored 0–5 by multiple commercial LLMs (GPT, Claude, Gemini, Grok). The QA score **correlates linearly with composite score (r = 0.94)**, so optimizing composite score during development is a reliable proxy for final performance.

**You do not need to run the GraphRAG QA pipeline yourself.** Organizers handle it — it costs ~$8–10 per run.

### Model Restriction

To ensure fair comparison and enable future **local deployment**, KG extraction may use only these OpenRouter models:

| Model | Notes |
|-------|-------|
| `z-ai/glm-4.7-flash` | Naive implementation default |
| `qwen/qwen3-14b` | |
| `nvidia/nemotron-3-nano-30b-a3b` | |
| `openai/gpt-oss-20b` | |
| `deepseek/deepseek-r1-distill-qwen-32b` | |

See `kg_extraction.py` for API usage reference.

## Quick Start

```bash
# Clone
git clone https://github.com/chicago-aiscience/Clinical_KG_OS_LLM.git
cd Clinical_KG_OS_LLM

# Install uv, then sync dependencies
# https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Step 1: API keys
cp api_keys_example.json api_keys.json
# Edit api_keys.json: replace the placeholder with your key from https://openrouter.ai/keys

# Step 2: KG extraction (runs the 6-pass multi-agent pipeline)
# Tip: test on a small subset first to iterate quickly
uv run python -m Clinical_KG_OS_LLM.kg_extraction --output ./my_kg --res-ids RES0198 RES0199
# Full run (all 20 transcripts, takes several minutes):
uv run python -m Clinical_KG_OS_LLM.kg_extraction --output ./my_kg

# Step 3: Entity resolution — merge per-patient KGs into a unified graph
uv run python -m Clinical_KG_OS_LLM.dump_graph --input ./my_kg --output ./my_kg_unified

# Step 4: Score your KG against the human-curated reference
uv run python -m Clinical_KG_OS_LLM.kg_similarity_scorer \
  --student ./my_kg_unified/unified_graph_my_kg.json \
  --baseline ./data/human_curated/unified_graph_curated.json

# Step 5: Visualize (optional)
uv run python -m Clinical_KG_OS_LLM.visualize_kg \
  --kg ./my_kg_unified/unified_graph_my_kg.json \
  --output ./my_kg_unified/kg_graph.png
```

> **Path note:** `dump_graph` writes `unified_graph_{input_dir_name}.json` — e.g. `unified_graph_my_kg.json` when `--input ./my_kg`.

**Optional — run GraphRAG QA locally:**

```bash
uv run python -m Clinical_KG_OS_LLM.graphrag_qa_pipeline \
  --kg ./my_kg_unified/unified_graph_my_kg.json
# Results written to ./my_kg_unified/results_unified_graph_my_kg/
```

### Notebook

Prefer a notebook? `notebooks/quickstart.ipynb` covers Steps 1–5 interactively.

```bash
uv sync --extra jupyter
jupyter lab
```

## Pipeline Overview

![Pipeline](figures/pipeline_overview.png)

### Stage 1: KG Construction

#### KG Extraction (`kg_extraction.py`)

The provided implementation runs a **6-pass multi-agent pipeline** per transcript:

| Pass | What happens |
|------|-------------|
| **1** | Two parallel node extraction agents; results merged on `(text, type)` to maximize recall |
| **2** | Assessment check — the doctor's final turn is information-dense; a separate agent adds any missed entities |
| **3** | Node review — tool-calling agent reads each node's source turn via `get_turn` / `search_transcript`, drops denied or unsupported nodes |
| **4** | Canonicalization — node text normalized to standard clinical form (e.g. `"liquid stools"` → `"diarrhea"`) |
| **5** | Batched edge extraction — two parallel agents per batch of source nodes; edges require explicit transcript evidence |
| **5a–5c** | Python schema filter → evidence filter → LLM clinical plausibility review |

The `--method` flag selects the extraction strategy:
- `node_edge` (default): the 6-pass pipeline above
- `naive`: single-pass, no verification

#### Entity Resolution (`dump_graph.py`)

Merges 20 per-patient KGs into a single unified graph:
- Embeds all node texts with [BGE-M3](https://arxiv.org/abs/2402.03216)
- Clusters by cosine similarity (threshold: **0.85**) within each node type
- Deduplicates: `"high blood pressure"` = `"hypertension"` → one canonical node

### Stage 2: Evaluation

| Script | Purpose |
|--------|---------|
| `graphrag_qa_pipeline.py` | Retrieves relevant KG triples for a clinical question and generates an answer |
| `kg_similarity_scorer.py` | Computes composite score against the human-curated reference KG |
| `llm_judge_batch_parallel.py` | Multi-LLM answer scoring — **run by organizers only, not provided** |

## Expected Output

Submit a single JSON file: `unified_graph_{your_name}.json` — a unified, deduplicated knowledge graph produced by running your extraction pipeline over the 20 transcripts and then `dump_graph.py`.

Steps:
1. Run your KG extraction pipeline → per-patient JSON files
2. Run `dump_graph.py` → unified graph
3. Optionally verify composite score with `kg_similarity_scorer.py`
4. Organizers run GraphRAG QA + LLM judge on your submission

## Project Structure

```
├── data/
│   ├── transcripts/               # 20 patient transcripts (+ .mp3 audio)
│   ├── human_curated/             # Human-curated reference KG (evaluation target)
│   └── naive_results/             # Pre-computed naive baseline results
├── src/Clinical_KG_OS_LLM/
│   ├── kg_extraction.py           # 6-pass multi-agent pipeline (reference implementation)
│   ├── dump_graph.py              # Entity resolution & KG merging (BGE-M3)
│   ├── graphrag_qa_pipeline.py    # GraphRAG QA pipeline
│   ├── kg_similarity_scorer.py    # Composite score against reference KG
│   └── visualize_kg.py            # KG visualization
├── notebooks/quickstart.ipynb     # Interactive walkthrough
├── figures/                       # Architecture and pipeline diagrams
└── pyproject.toml                 # Dependencies (use `uv sync`)
```

## Architecture Directions

![Multi-Agent Architectures](figures/multi-agent-architectures.png)

## Additional Challenge: Speech-to-Text

Each patient folder includes the original `.mp3` recording. Pre-generated transcripts are provided, but teams can optionally build a pipeline that goes directly from `.mp3` to Knowledge Graph.

**Open-source SOTA:** [OpenAI Whisper](https://github.com/openai/whisper) ([Radford et al. 2022](https://arxiv.org/abs/2212.04356))

```bash
pip install openai-whisper
conda install ffmpeg
```

| Method | WER | Accuracy |
|--------|-----|----------|
| Whisper base (baseline) | 83.3% | 16.7% |
| This pipeline | **16.7%** | **83.3%** |
