# Project Literature Triage

A small CrewAI project you can copy and build on. It answers one question about a local set of papers. A router reads the question and chooses which agents run. The screener always reads the abstracts in `papers/`. Later agents read the full text in `fulltext/`. The writer always runs last. They call a model on your own machine through [Ollama](https://ollama.com). No cloud API key is required.

```text
papers/                   # one Markdown abstract per file
fulltext/                 # matching full text, same file name
starter/
  setup.py                # starts Ollama and downloads the model
  main.py                 # loads both folders, routes, and runs the crew
  router.py               # keyword checks that choose the agents
  single_main.py          # the same first four steps, one agent, no router
  llm.py                  # the only place the model is chosen
  crew.py                 # wires agents, tasks, and tools
  config/agents.yaml      # add a role here
  config/tasks.yaml       # add a step here
  tools/example_tool.py   # copy this when you add a tool
```

`{question}`, `{abstracts}`, `{fulltext}`, and `{route}` in the YAML files are filled in when you run the crew.

## What you need

- Python 3.10, 3.11, 3.12, or 3.13
- About 8 GB of free disk for a small model, more for larger ones
- [Ollama](https://ollama.com) installed and able to reach `http://localhost:11434`

`llama3.1:8b` is the default. It is large enough to follow the four-agent handoff. Smaller models often ignore the task format.

## 1. Install Ollama

**macOS:** download the app from [ollama.com](https://ollama.com), or:

```bash
brew install ollama
```

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** download the installer from [ollama.com](https://ollama.com).

You only need the Ollama program installed in this step. The next step starts it and downloads the model. Do not run `ollama pull` yourself: if your shell has `OLLAMA_HOST` pointed at a port where nothing is listening, that command fails even when Ollama is already running.

## 2. Set up this project

From the project folder, check the Python version first. It must be 3.10 or newer. Python 3.8 and 3.9 cannot install current CrewAI.

```bash
python3 --version
python3 -m venv .venv
```

If that `python3` is too old, install Python 3.12 and create the environment with it, for example `python3.12 -m venv .venv`.

Activate the virtual environment:

```bash
# macOS and Linux
source .venv/bin/activate

# Windows Command Prompt
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies. This pulls CrewAI and LiteLLM, which routes `ollama/...` model names to your local server:

```bash
pip install -r requirements.txt
```

Copy the environment file and leave the defaults if you pulled `llama3.1:8b`:

```bash
cp .env.example .env
```

On Windows Command Prompt use `copy .env.example .env`.

Start Ollama if it is stopped, and download the model named in `.env`:

```bash
python -m starter.setup
```

This looks for a server on `http://127.0.0.1:11434` and on `OLLAMA_HOST`. If one is already running, the download uses that server. If none is running, it starts Ollama on port `11434` and downloads there. A stale `OLLAMA_HOST` in `~/.bashrc` cannot send the download to the wrong port.

The first download of `llama3.1:8b` is several gigabytes. When the command prints `Ready`, continue to the next step.

## 3. Run it

```bash
python -m starter.main
```

That loads every Markdown file in `papers/` and `fulltext/`, then asks the default question. Pass your own question as an argument. It replaces `{question}` in the agent and task files:

```bash
python -m starter.main "Which papers used a Transformer architecture and what task did they evaluate it on?"
```

The router prints the route before the crew starts. Relevance always runs and sees only `papers/`. Synthesis always runs last. The steps in between depend on the wording:

| Route | The question contains | Then these agents run |
| --- | --- | --- |
| disagreement | disagree, agree, or contradict | extraction, contradiction, synthesis |
| coverage | beginner, which 5, which five, or read first | coverage, synthesis |
| survey | summarize or main approaches | extraction, synthesis |
| evidence | provide evidence | extraction, contradiction, synthesis |
| lookup | transformer, architecture, or which papers used | extraction, synthesis |

The first matching row wins. Any other question takes the evidence route. The checks live in `starter/router.py`.

`papers/` is one abstract per file. `fulltext/` is the full text of the same paper, with the same file name. Later agents receive every file in `fulltext/`. That folder is about 800,000 characters, which is more than `llama3.1:8b` can read in one turn, so a later step can fail even when the screener finishes.

To run the same screening, extraction, contradiction, and writing steps with one agent and no router:

```bash
python -m starter.single_main "Which papers used a Transformer architecture and what task did they evaluate it on?"
```

To change the default question without passing an argument, edit `QUESTION` in `.env`. More questions are listed in `questions.txt`.

The first run is slow because the model loads into memory.

## 4. Point it at a different model

Edit `.env`:

```bash
OLLAMA_MODEL=qwen2.5:7b
```

Download that model with the same setup command. It reads `OLLAMA_MODEL` from `.env`:

```bash
python -m starter.setup
```

`qwen2.5:7b` is a good next step when you turn on tools. Keep `LLM_TEMPERATURE` around `0.2` so the agents stay on the brief.

If Ollama runs on another computer, set the host and start Ollama so it listens on the network:

```bash
OLLAMA_BASE_URL=http://192.168.1.20:11434
```

On the Ollama machine:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

## The handoff

The screener does not see the full text. Later agents do not see the abstracts as their source.

| Agent | Reads | Task |
| --- | --- | --- |
| Relevance | The question and every abstract | KEEP or DISCARD, plus a reason |
| Extraction | That decision list, then the full text | Method, dataset, and result for kept papers only |
| Contradiction | The extracted notes and the full text | Places two papers disagree, with both claims |
| Coverage | That decision list and the full text | Which kept papers show each reading-list dimension |
| Synthesis | The notes from the steps that ran, and the full text | The answer, with a paper title on every claim |

The disagreement rule sits on the synthesis task: if the contradiction step ran and flagged a conflict, the answer states both sides and cites both papers. It does not drop one side to sound certain. On a lookup or survey route the contradiction step does not run.

`context` in `starter/config/tasks.yaml` is the handoff. A task listed there receives the earlier task's output. `starter/router.py` chooses which of those tasks run. Relevance is first and synthesis is last.

## 6. Add an agent

1. Add a block to `starter/config/agents.yaml`. The top-level name is the method name you will add in Python. Set `llm: local` so it uses the same model.

```yaml
editor:
  role: Editor
  goal: Tighten the answer to {question} without adding new claims
  backstory: You cut repetition and flag anything the earlier notes do not support.
  llm: local
  verbose: true
  max_iter: 5
```

2. Add a method in `starter/crew.py` with the same name:

```python
@agent
def editor(self) -> Agent:
    return Agent(config=self.agents_config["editor"])
```

3. Give that agent a task (next section). An agent with no task does not run.

## 7. Add a task

1. Add a block to `starter/config/tasks.yaml`. `agent` must match an `@agent` method. `context` lists earlier tasks whose output this step should see.

```yaml
edit_task:
  description: Edit the answer to {question}. Keep every claim tied to the notes above.
  expected_output: The revised answer, plus a short list of cuts you made.
  agent: editor
  context:
    - synthesis_task
```

2. Add the matching method in `starter/crew.py`:

```python
@task
def edit_task(self) -> Task:
    return Task(config=self.tasks_config["edit_task"])
```

A task runs only when `starter/router.py` includes it for that question. Add the task name to the matching route there, or it stays unused. Relevance still runs first and synthesis still runs last.

## 8. Add a tool

The example tool counts words. It stays off until you opt in, because small local models often mishandle tool calls.

1. In `starter/config/agents.yaml`, uncomment the tool on the agent that should use it:

```yaml
tools:
  - word_count
```

2. Run the crew again. `word_count` is already defined in `starter/crew.py` and implemented in `starter/tools/example_tool.py`.

To add your own tool, copy `starter/tools/example_tool.py`, change the class, then expose it from the crew:

```python
@tool
def my_tool(self):
    return MyTool()
```

Reference `my_tool` under `tools:` in the agent YAML. The YAML name and the method name must match.

## If it fails

| What you see | What to do |
| --- | --- |
| `could not connect to ollama server` after `ollama pull` | Run `python -m starter.setup` instead. Your shell's `OLLAMA_HOST` is pointed at a different port than the running server. |
| `Ollama is not running` | Run `python -m starter.setup`. It starts the server on port 11434. |
| Model is not installed | Run `python -m starter.setup`, or set `OLLAMA_MODEL` in `.env` to a model you already have and run setup again. |
| `OPENAI_API_KEY` error | Copy `.env.example` to `.env`. The Ollama path does not call OpenAI; CrewAI still wants a placeholder key when you switch providers. |
| Agent loops or ignores the format | Stay on `llama3.1:8b` or larger, keep temperature at `0.2`, and leave tools commented out until the brief looks right. |
| Connection error to another machine | Use that machine's real IP in `OLLAMA_BASE_URL`, and start Ollama with `OLLAMA_HOST=0.0.0.0:11434`. |

## Project layout

| File | Role |
| --- | --- |
| `starter/setup.py` | Starts Ollama and downloads the model in `.env` |
| `starter/ollama_server.py` | Finds the running server and ignores a stale `OLLAMA_HOST` |
| `papers/` | One Markdown abstract per file. The screener reads this folder. |
| `fulltext/` | Full text for the same file names. Later agents read this folder. |
| `questions.txt` | Sample questions, one type per heading |
| `starter/main.py` | Loads `.env`, reads both folders, routes the question, runs the crew |
| `starter/router.py` | Maps the question text to lookup, survey, evidence, disagreement, or coverage |
| `starter/single_main.py` | Runs one agent through screening, extraction, contradiction, and writing |
| `starter/llm.py` | Builds the local LLM from `.env` |
| `starter/crew.py` | Connects YAML names to Python methods and keeps only the routed tasks |
| `starter/config/agents.yaml` | Roles, goals, model, optional tools |
| `starter/config/tasks.yaml` | Steps, expected output, and which agent does each step |
| `starter/tools/example_tool.py` | Pattern for a custom tool |
