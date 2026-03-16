# data_synthesis

End-to-end pipeline for synthesising search-oriented QA pairs from raw trajectory data.

---

## Pipeline Overview

```
Input JSONL (raw trajectories)
        │
        ▼
step1_extract_urls_from_traj.py – Clean & normalise trajectory records
        │
        ▼
step2_broaden_information.py   – Enrich entities with broader context (LLM)
        │
        ▼
step3_construct_graph_stage_a.py – Extract entities & relations, build raw KG (LightRAG)
        │
        ▼
step3_construct_graph_stage_b.py – Merge / deduplicate graph nodes
        │
        ▼
step4_visualize_graph.py       – (Optional) Render the full knowledge graph
        │
        ▼
step5_extract_subgraph.py      – Extract target subgraphs + factuality filtering (LLM)
        │
        ▼
step6_visualize_subgraph.py    – (Optional) Render individual subgraphs
        │
        ▼
step7_layerwise_description_generator.py
                           – Layer-wise question generation with iterative
                             verification & obfuscation (LLM)
        │
        ▼
step8_get_qa.py                – Extract final QA pairs + statistics
        │
        ▼
Output: <stem>_qa.jsonl  +  <stem>_qa_stats.json
```

---

## Quick Start

```bash
# Run the full pipeline (bash / WSL / Git-Bash)
INPUT_FILE=/path/to/raw_data.jsonl \
OUTPUT_DIR=./results \
GENERATE_MODEL=deepseek-v3.2 \
VERIFY_MODEL=gpt-5-mini \
PARALLEL=4 \
bash data_synthesis/run_pipeline.sh
```

Or run each step individually (see per-script `--help` for all options):

```bash
python data_synthesis/1_extract_urls_from_traj.py --input raw.jsonl --output step1.jsonl
python data_synthesis/2_broaden_information.py --input step1.jsonl --output step2.jsonl
# … and so on
```

---

## Environment Variables (run_pipeline.sh)

| Variable | Default | Description |
|---|---|---|
| `INPUT_FILE` | *(required)* | Path to the raw trajectory JSONL |
| `OUTPUT_DIR` | `./results` | Directory for intermediate & final outputs |
| `GENERATE_MODEL` | `deepseek-v3.2` | LLM used for description/question generation |
| `VERIFY_MODEL` | `gpt-5-mini` | LLM used for verification & obfuscation |
| `PARALLEL` | `1` | Number of concurrent workers (Step 7) |
| `MAX_OBF_ITER` | `5` | Maximum obfuscation iterations per node (Step 7) |

---

## File Descriptions

| File | Description |
|---|---|
| `utils.py` | Shared utilities: JSONL I/O, LLM client, string helpers |
| `step1_extract_urls_from_traj.py` | Trajectory normalisation |
| `step2_broaden_information.py` | Entity context enrichment via LLM |
| `step3_construct_graph_stage_a.py` | KG construction using LightRAG |
| `step3_construct_graph_stage_b.py` | Graph deduplication & merging |
| `step4_visualize_graph.py` | Full-graph visualisation (matplotlib) |
| `step5_extract_subgraph.py` | Subgraph extraction + LLM factuality scoring |
| `step6_visualize_subgraph.py` | Per-subgraph visualisation (matplotlib) |
| `step7_layerwise_description_generator.py` | Layer-wise question generation with obfuscation |
| `step8_get_qa.py` | QA pair extraction & statistics |
| `run_pipeline.sh` | One-command pipeline runner |

---

## Output Format

### QA pair (`*_qa.jsonl`)

Each line is a JSON object with the following fields:

```jsonc
{
  "question": "...",          // generated search-oriented question
  "golden_answer": "...",     // entity name (ground-truth answer)
  "graph_id": "...",
  "subgraph_id": "...",
  "root_entity": "...",
  "topology": { ... },        // subgraph topology metadata
  "obfuscation_iterations": 2 // number of obfuscation rounds applied
}
```

### Statistics (`*_qa_stats.json`)

Summary counts for total graphs, subgraphs, QA pairs, obfuscation distribution, node-type distribution, and depth distribution.

---

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Key libraries:

- `networkx` – graph construction and analysis
- `lightrag` – KG extraction framework
- `matplotlib` – visualisation
- `openai` / compatible SDK – LLM calls
- `asyncio`, `concurrent.futures` – concurrency
