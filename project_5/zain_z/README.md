# Project 5: Missing-metadata agent

An agent that fills in missing metadata for a Hugging Face model. For every property it
fills, it records **where the value came from and the exact sentence that states it**.
Properties it cannot back with a quote are reported as unknown or flagged for review.

Built with [CrewAI](https://docs.crewai.com) on a local LLM (served by vLLM).

For a step-by-step look at a real run, see [docs/walkthrough.md](docs/walkthrough.md).

## What it does

Input: a Hugging Face model id, e.g. `microsoft/phi-1_5`.

Output: `output/microsoft__phi-1_5.json`, with one entry per property. Example:

```json
"parameterCount": {
    "value": "1.42B",
    "status": "found",
    "evidence": [
        {
            "source": "metadata",
            "url": "https://huggingface.co/microsoft/phi-1_5",
            "quote": "safetensors_total_params: 1418270720"
        }
    ],
    "notes": ""
}
```

The properties it extracts:

| Property | Meaning |
|---|---|
| `mlTask` | The task the model performs (Hugging Face pipeline tag) |
| `modelArchitecture` | The architecture, as the source names it |
| `parameterCount` | Total parameters of this exact model |
| `baseModel` | The model it was fine-tuned from, or `none` if it was trained from scratch |
| `trainingData` | The datasets it was trained on |
| `trainingTokens` | Total tokens seen during training |

### Status

| Status | Meaning |
|---|---|
| `found` | A source states the value directly, about this model. |
| `review` | A value was found, but no source states it directly (e.g. it is only implied, or had to be calculated). A human should check it. |
| `unknown` | No source gave a value. |

`none` is a **value**, not a status: `baseModel: "none"` with status `found` means a source
says the model is a base model itself.

## How it works

The model's information comes from three sources, from cheapest to most expensive:

1. **metadata**: structured Hugging Face metadata (pipeline tag, config, parameter count, base model, dataset tags, linked arXiv ids)
2. **card**: the model card (the repo's README.md)
3. **paper**: the paper that introduces the model, found through the repo or an arXiv search

Each property is extracted in its own run, so every run starts with a small, clean context:

```
for each property:
    already in output/<model>.json with a value?  -> skip
    extract task     check the allowed sources in order; stop at the first one that
                     states the value directly (confident = true)
    reference task   does any part of the answer only point to another model?
                     (e.g. "phi-1's training data")
    if so            run the same extraction for that model, from its paper,
                     and replace the pointer with what it finds (one level deep)
    write the entry to output/<model>.json
```

### The extract task

One agent gets the property, a one-line description of it (the hint), and the tools of
the sources it may use, in order. After each source it decides:

- the source **states the value directly**, it is **about this exact model**, and it
  **does not just point to another model**: stop and answer with `confident = true`;
- otherwise, move on to the next source.

If no source is confident, it returns the best value it found with `confident = false`
(status `review`), or no value (status `unknown`).

The answer is structured (a Pydantic model): the `value` in as few words as possible, and
`evidence`, a list of `{source, url, quote}` where every quote is one exact passage from a
tool output.

### The reference task

Sources often describe a property by pointing to another model: the phi-1.5 paper says its
training data is "a combination of phi-1's training data and ...". The extract agent tends
to accept "phi-1's training data" as a dataset name, even when told not to.

So a second, small task only classifies: for each item of the answer, does it point to
another model's value instead of stating one? This task gets only the extract task's
answer, not the long source texts, so the question stays in focus. For every pointer,
the code runs the whole extraction again for the referenced model (paper only) and merges
the result into the answer, keeping the evidence of both.

## Project structure

```
project_5/
├── main.py              entry point: property loop, skip existing answers, output JSON, logging
├── crew.py              agent, tasks, output models, reference resolution (extract, merge)
├── tools.py             the tools and the .cache
├── config/
│   ├── agents.yaml      the agent (role, goal, backstory, max_iter)
│   └── tasks.yaml       extract and reference task prompts, source descriptions, property hints
├── data/
│   ├── properties.json  the properties to extract and the sources each may use, in order
│   └── hf_models.json   hand-made ground truth (phi-1.5)
├── docs/
│   └── walkthrough.md   an annotated real run
├── output/              one <org>__<model>.json per model (created on first run)
├── .cache/              fetched metadata, cards and papers (created on first run)
└── .log/                one log file per run (created on first run)
```

### Tools

| Tool | Source | What it returns |
|---|---|---|
| `hf_metadata` | metadata | pipeline tag, architecture class, exact parameter count, base model, dataset tags, arXiv ids |
| `hf_modelcard` | card | the full model card text |
| `fetch_paper_local` | paper | candidate papers the repo mentions: arXiv tags, arXiv links and BibTeX titles in the card |
| `fetch_paper_api` | paper | arXiv search results (id, title, abstract) for free text, e.g. a model name or paper title |
| `retrieve_paper_from_arxiv` | paper | the full text of an arXiv paper (PDF to text) |

Every fetch except the arXiv search is cached in `.cache/` (metadata, cards, paper
candidates, paper texts), so running several properties or models does not download the
same card or paper twice.

## Setup

Requirements: Python 3.12 and an OpenAI-compatible LLM endpoint with tool calling.
We serve `cyankiwi/Qwen3.8-27B-AWQ-INT4` (4-bit, 21 GB) with vLLM on one RTX A6000 (48 GB),
with a 131k context so that two full papers fit in one run:

```
sudo docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8000:8000 --ipc=host \
  vllm/vllm-openai:v0.30.0 \
  --model cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.94 --enable-prefix-caching \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

The reasoning parser is required for tool calling with this model. `--language-model-only`
skips the model's vision part.

```
pip install -r requirements.txt
```

Create `project_5/.env`:

```
# Any LiteLLM model string: openai/<name> for vLLM or LM Studio, ollama/<name> for Ollama
LLM_MODEL=openai/cyankiwi/Qwen3.8-27B-AWQ-INT4
LLM_BASE_URL=http://localhost:8001/v1
LLM_API_KEY=local
LLM_TEMPERATURE=1.0
```

Qwen3.8 thinks before it answers; its model card recommends temperature 1.0 for that mode.

### Model choice

We first used `cyankiwi/gemma-4-31B-it-qat-AWQ-INT4` (temperature 0.2) and compared it with
Qwen3.8 on `microsoft/phi-1_5` and `google/flan-t5-base`, with the same prompts:

| | Gemma 4 31B | Qwen3.8 27B |
|---|---|---|
| phi-1.5 `trainingData` | descriptions ("filtered code-language dataset") | real names (The Stack, StackOverflow, ...), matching the ground truth |
| Following references (phi-1.5 to phi-1) | yes | yes, with more evidence |
| `modelArchitecture` | from the card or paper | took `model_type` from the metadata until the hint excluded it |
| Time per model (6 properties) | 3 to 5 min | about 8 min |

Most properties came out identical. Qwen's thinking pays off on the hardest property
(following a reference to another model's paper); its slips were prompt gaps that a hint
fixed. It is about twice as slow.

## Running

```
python main.py microsoft/phi-1_5                          # all properties
python main.py microsoft/phi-1_5 trainingData baseModel   # only these
```

- Results go to `output/<org>__<model>.json`, written after every property, so a crash
  keeps what was already found.
- Properties that already have a value in that file are skipped. Only missing or `unknown`
  ones are extracted. To redo one, delete its entry (or the file).
- Everything printed during a run is also saved to `.log/<date>_<time>.log`.
- To fetch sources again (e.g. a card was updated), delete `.cache/` or the file in it.

## Configuration

**`data/properties.json`** lists the properties and the sources each may use, in the order
they are checked. A source that can never hold a property is left out, so the agent does
not waste time on it (`trainingTokens` is never in the metadata; `mlTask` never needs a paper):

```json
"mlTask": ["metadata", "card"],
"trainingTokens": ["card", "paper"]
```

**`config/tasks.yaml`** holds the prompts. `hints` has one short description per
property. Hints describe what to extract, not which answers are allowed: example values
in a prompt make the model force its answer into them.

### Adding a property

1. Add it to `data/properties.json` with its allowed sources.
2. Add a hint for it under `hints` in `config/tasks.yaml`.

## Known limitations

Some of the limitations we came across during the test run:

- **Incomplete dataset lists in metadata.** A `datasets` list in the metadata is taken as
  confident, so the card and paper are not read. For `google/flan-t5-base` the metadata
  lists 10 datasets out of the 1,800+ tasks it was fine-tuned on. Authors usually list the
  datasets they want to report, so this is accepted.
- **Summarised references are not followed.** "The same data as phi-1, a mix of code and
  text" gives the value `[code, text]`, which does not point anywhere, so phi-1's real
  dataset list is never looked up.
- **Referenced models are looked up by name, from their paper only.** The reference task
  returns a model name ("phi-1"), not a repo id, so the referenced model's metadata and
  card are not used. The agent sometimes guesses the repo id itself (`microsoft/phi-1`),
  which is not grounded.
- **Papers about several models.** A paper covering a model family can mislead attribution:
  for `google/flan-t5-base`, `trainingTokens` came from sentences about Flan-PaLM (Gemma) or
  about the base T5 model (Qwen). Both were flagged `review`, not `found`.
- **Very long papers.** Papers are passed whole. The Flan paper (`google/flan-t5-base`)
  lists 1,800+ tasks in its appendix, and Qwen retrieved it twice in one run, which went
  over the 131k context. Qwen sometimes repeats the same tool call, which doubles the text.
- **Runs are not deterministic.** At temperature 1.0 the same prompt can give a different
  format: in one phi-1.5 run the training data came back as one string instead of a list,
  so the reference to phi-1 was not followed (status `review`). Re-running that property fixed it.
- **Only arXiv papers** are found and read.
