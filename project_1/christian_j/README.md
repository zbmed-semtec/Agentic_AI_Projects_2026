# Card Guide – Simple Question-Answering Agent

A terminal chat that answers questions about a fixed collection of machine-learning
model cards and then suggests follow-up questions. The answer comes only from the
cards, names the file(s) it used, and says so when the collection has no answer.

Built with [CrewAI](https://github.com/crewAIInc/crewAI) and a local LLM via
[Ollama](https://ollama.com).

## How it works

Two agents run one after the other (`Process.sequential`) for every question:

| # | Agent | Task | Tools | Output |
|---|---|---|---|---|
| 1 | **Model Card Assistant** (`card_guide`) | `answer_question` | `list_cards`, `search_cards`, `read_card` | The answer with a `Source:` line |
| 2 | **Follow-up Question Proposer** (`suggestion_helper`) | `suggest_questions` | `list_cards` | 3 follow-up questions |

**Agent 1** decides step by step which tool to use: `list_cards` for name, license
or task questions, `search_cards` for anything in the card text, and `read_card`
only when it needs details from one specific card. It stops as soon as the tool
results answer the question. The loop is capped by `max_iter`.

**Agent 2** receives the first answer through `context: [answer_question]` in
`tasks.yaml`. It checks with `list_cards` which models exist and suggests 3
questions the collection can actually answer. If agent 1 found nothing, it
suggests questions about models that are in the collection instead.

`main.py` prints both results separately:

```
Answer: Phi uses the MIT license.
Source: Microsoft__phi.md

Suggestions:
1. Which other models use the MIT license?
2. Which models are for text generation?
3. ...
```

## Project structure

```
Makefile             # shortcuts: make setup / run / check / pull / serve / cards / lint
config/
  agents.yaml        # both agents: role, goal, instructions, tools, LLM
  tasks.yaml         # both tasks: description, expected output, context
data/
  models.json        # raw source data
  model_cards/       # 20 model cards as Markdown (generated)
scripts/
  main.py            # terminal chat loop (entry point)
  check_ollama.py    # checks that Ollama runs and the model is installed
  crew.py            # CrewAI crew: wires YAML config, LLM and tools together
  llm.py             # builds the LLM from .env
  build_cards.py     # generates data/model_cards/ from data/models.json
tools/
  list_cards.py      # list_cards: overview of all cards
  search_cards.py    # search_cards: keyword search in the full card text
  read_card.py       # read_card: opens one card by filename
```

### Tools

| Tool | Purpose |
|---|---|
| `list_cards` | Lists every card with filename, model name, license and tasks. |
| `search_cards` | Searches all cards for one keyword (at word start). Returns filename + the text around each match, and a last line `Found in N cards: ...`. |
| `read_card` | Opens one card by its filename (e.g. `Microsoft__phi.md`) and returns its full text. |

## Setup

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), Ollama, `make`.

1. Create your `.env` from the template (the default model is `qwen2.5:7b`):

```bash
cp .env.example .env
```

2. Install dependencies and download the model named in `.env`:

```bash
make setup
```

To use another model, change `OLLAMA_MODEL` in `.env` and run `make pull`.
Pick a model with tool support (marked "tools" on [ollama.com](https://ollama.com/search?c=tools)).
Run `make` without a target to list all commands.

## Usage

Start the chat **from the project root**:

```bash
make run
```

Before the chat starts, it checks that Ollama is answering and the model is
installed. If not, it stops with a hint; `make serve` starts Ollama and
`make pull` downloads the model.
You can run this check on its own with `make check`.

Type a question and press Enter. An empty line quits.
With `verbose: true` the terminal shows every step: which tool each agent calls
and what it reads.

> Run it from the project root: `python -m scripts.main` and the `.env` file are
> both looked up from the current folder. `make run` takes care of that.

### Sample questions

- What license does Phi use?
- Which models are for translating text?
- What license does bert-base-uncased use?
- Which models mention bias or limitations?
- Find a model that works with images.

Answers end with a `Source:` line naming the card file(s). If no card contains the
answer, the agent replies: *"I could not find relevant cards for this question."*

## Rebuilding the dataset

```bash
make cards
```

Picks 20 entries from `data/models.json` with enough description text, spread over
different tasks, and writes them to `data/model_cards/`.

## Development

```bash
make lint
```

Checks and formats the Python code with [ruff](https://docs.astral.sh/ruff/).
`data/` is excluded so the model cards stay unchanged.
