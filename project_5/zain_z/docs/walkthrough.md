# Walkthrough: one run on phi-1.5

A real run of the agent on [`microsoft/phi-1_5`](https://huggingface.co/microsoft/phi-1_5),
with excerpts from its log (`.log/2026-09-30_11-29-59.log`, LLM: Gemma 4 31B, about 2 minutes
for all six properties). Excerpts are trimmed.

```
python main.py microsoft/phi-1_5
```

For every property, the agent checks the sources in order (**metadata → card → paper**)
and stops at the first one that states the value directly (`confident: true`, written as
status `found`). A second, small task then checks whether the answer only points to
another model.

## 1. `mlTask`

**Sources allowed:** metadata, card

The agent calls the first source:

```
Tool: hf_metadata   Args: {'repo_id': 'microsoft/phi-1_5'}
Output: {"pipeline_tag": "text-generation", "architectures": ["PhiForCausalLM"],
         "safetensors_total_params": 1418270720, "base_model": null,
         "datasets": null, "arxiv_papers": ["2309.05463"], ...}
```

`pipeline_tag` states the value directly, so it stops after one call:

```json
{
  "value": "text-generation",
  "evidence": [{"source": "metadata", "quote": "\"pipeline_tag\": \"text-generation\""}],
  "confident": true
}
```

**Result:** `text-generation`, found, from the metadata.

## 2. `modelArchitecture`

**Sources allowed:** metadata, card, paper

The metadata only has the class name `PhiForCausalLM`, and the prompt says a class name
does not state the architecture. So the agent moves on to the card:

```
Tool: hf_metadata    Args: {'repo_id': 'microsoft/phi-1_5'}
Tool: hf_modelcard   Args: {'repo_id': 'microsoft/phi-1_5'}
```

The card says it directly, so the paper is not needed:

```json
{
  "value": "Transformer-based model",
  "evidence": [{"source": "card",
                "quote": "Architecture: a Transformer-based model with next-word prediction objective"}],
  "confident": true
}
```

**Result:** `Transformer-based model`, found, from the card.

## 3. `parameterCount`

**Sources allowed:** metadata, card, paper

```
Tool: hf_metadata   Args: {'repo_id': 'microsoft/phi-1_5'}
```

The metadata has the exact count from the model's weight files. The card and paper only
give a rounded figure ("1.3 billion"), and the prompt asks to prefer the exact count:

```json
{
  "value": "1.42B",
  "evidence": [{"source": "metadata", "quote": "safetensors_total_params: 1418270720"}],
  "confident": true
}
```

**Result:** `1.42B`, found, from the metadata.

## 4. `baseModel`

**Sources allowed:** metadata, card, paper

The metadata has `"base_model": null`. That does not prove the model has no base model
(the field is often just not filled in), so the agent checks the card:

```
Tool: hf_metadata    Args: {'repo_id': 'microsoft/phi-1_5'}
Tool: hf_modelcard   Args: {'repo_id': 'microsoft/phi-1_5'}
```

```json
{
  "value": "none",
  "evidence": [{"source": "card",
                "quote": "Note that Phi-1.5, being a base model, often produces irrelevant text following the main answer."}],
  "confident": true
}
```

**Result:** `none`, found, from the card. "none" is a real answer here: a source says the
model has no base model. If nothing had been found, the status would be `unknown`.

## 5. `trainingData`

**Sources allowed:** metadata, card, paper

Neither the metadata (`"datasets": null`) nor the card name any datasets, so the agent
reads the paper:

```
Tool: hf_metadata                 Args: {'repo_id': 'microsoft/phi-1_5'}
Tool: hf_modelcard                Args: {'repo_id': 'microsoft/phi-1_5'}
Tool: retrieve_paper_from_arxiv   Args: {'arxiv_id': '2309.05463'}
```

The paper only says the data is "a combination of phi-1's training data and newly created
synthetic textbook-like data". The agent's answer contains that reference as a dataset:

```json
{
  "value": ["phi-1 training data", "synthetic, textbook-like data"],
  "evidence": [{"source": "paper", "url": "https://arxiv.org/abs/2309.05463",
                "quote": "Our training data for phi-1.5 is a combination of phi-1's training data
                          (7B tokens) and newly created synthetic, “textbook-like” data [...]"}]
}
```

**The reference task catches it.** It only sees this answer and asks, for each item,
whether it just points to another model:

```
references=[Reference(item='phi-1 training data', model='phi-1')]
Resolving 'phi-1 training data' via phi-1
```

**A second extraction runs for phi-1**, from its paper:

```
Tool: fetch_paper_local           Args: {'repo_id': 'microsoft/phi-1'}
Tool: retrieve_paper_from_arxiv   Args: {'arxiv_id': '2306.11644'}
```

```json
{
  "value": ["filtered code-language dataset", "synthetic textbook dataset", "synthetic exercises dataset"],
  "evidence": [{"source": "paper", "url": "https://arxiv.org/abs/2306.11644",
                "quote": "Our training relies on three main datasets: • A filtered code-language
                          dataset, which is a subset of The Stack and StackOverflow [...]
                          • A synthetic textbook dataset [...] • A small synthetic exercises dataset [...]"}],
  "confident": true
}
```

**The results are merged.** The pointer is replaced by phi-1's datasets, and the evidence
from both papers is kept.

**Result:** filtered code-language dataset, synthetic textbook dataset, synthetic exercises
dataset, synthetic textbook-like data; found, from the card, the phi-1.5 paper and the
phi-1 paper.

## 6. `trainingTokens`

**Sources allowed:** card, paper (the metadata never has it, so the agent does not get that tool)

```
Tool: hf_modelcard   Args: {'repo_id': 'microsoft/phi-1_5'}
```

The card lists two numbers next to each other:

```
* Dataset size: 30B tokens
* Training tokens: 150B tokens
```

The prompt asks for the tokens seen during training, not the dataset size, and the agent
picks the right one:

```json
{
  "value": "150B tokens",
  "evidence": [{"source": "card", "quote": "Training tokens: 150B tokens"}],
  "confident": true
}
```

**Result:** `150B tokens`, found, from the card.

## The result

`output/microsoft__phi-1_5.json`:

| Property | Value | Status | Source |
|---|---|---|---|
| `mlTask` | text-generation | found | metadata |
| `modelArchitecture` | Transformer-based model | found | card |
| `parameterCount` | 1.42B | found | metadata |
| `baseModel` | none | found | card |
| `trainingData` | filtered code-language dataset, synthetic textbook dataset, synthetic exercises dataset, synthetic textbook-like data | found | card, phi-1.5 paper, **phi-1 paper** |
| `trainingTokens` | 150B tokens | found | card |

Every value comes with the exact quote it was taken from. The cheap sources answered most
properties; only `trainingData` needed papers, including a second model's paper.
