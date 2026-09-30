# t5-small-indian-news

- Model ID: adi2606/t5-small-indian-news
- License: CC0: Public Domain
- Tasks: india, beginner, nlp, text, news, summarization, english, t5
- Frameworks: Transformers
- Shared by: ADITYA SINGH
- Source: https://www.kaggle.com/models/adi2606/t5-small-indian-news

## Description

### Model Summary 📊

**T5-Small-Indian-News** is a fine-tuned **T5 model** for summarizing Indian news articles. It processes long-form text and generates concise summaries while retaining the core message.

### Usage ⚙️

The model summarizes news articles, providing quick insights for readers. It’s ideal for news summarization tools, content recommendations, and more.

### System 🖥️

This model is part of a **news summarization system**, taking articles as input and generating summaries as output. It requires text input and provides short, meaningful summaries.

### Implementation Requirements 💻

- **Hardware**: GPU (recommended for faster inference)
- **Software**: Hugging Face's `transformers`, PyTorch

### Model Characteristics 🧠

- **Fine-tuned from**: T5-small
- **Size**: ~1.7 GB
- **Latency**: Fast on GPUs

### Data Overview 📚

Trained on diverse **Indian news articles**, covering politics, economy, tech, and more. Pre-processed for tokenization and cleaning.

### Evaluation Results 📈

- **Performance**: High-quality summaries with good ROUGE scores
- **Limitations**: May struggle with highly specialized topics or sensitive issues.

### Ethics ⚖️

Ethical considerations include avoiding harmful or biased content. The model may require adjustments for sensitive topics.

## References

### 🧬 Provenance

This model is fine-tuned from the original **T5-small** model provided by Hugging Face.  
The fine-tuning was performed on a curated dataset of Indian news articles, focusing on improving summarization quality specific to regional content.

The dataset was preprocessed and tokenized using the T5 tokenizer. Training was conducted using PyTorch on a single NVIDIA GPU over multiple epochs, with careful hyperparameter tuning.

This version improves upon the base model by adapting to Indian English news style, sentence structure, and commonly used expressions.
