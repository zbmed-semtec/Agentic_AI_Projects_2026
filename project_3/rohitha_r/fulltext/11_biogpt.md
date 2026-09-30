Fetched from: https://ar5iv.labs.arxiv.org/html/2210.10341

\appnotes Published at Briefings in Bioinformatics

\authormark Renqian Luo et al.

\corresp [∗]Corresponding authors: Renqian Luo, Microsoft Research AI4Science, E-mail: renqianluo@microsoft.com; Yingce Xia, Microsoft Research AI4Science, E-mail: yinxia@microsoft.com; Tao Qin, Microsoft Research AI4Science, E-mail: taoqin@microsoft.com

# BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining

Renqian Luo\ORCID0000-0002-9062-34841,∗ Liai Sun2 Yingce Xia\ORCID0000-0001-9823-90331,∗ Tao Qin\ORCID0000-0002-9095-07761,∗ Sheng Zhang\ORCID0000-0003-3672-54363 Hoifung Poon\ORCID0000-0002-9067-09183 Tie-Yan Liu1 Address: Microsoft Research AI4Science, Beijing, China Address: Peking University, Beijing, China Address: Microsoft Research, Redmond, WA, USA Abstract Pre-trained language models have attracted increasing attention in the biomedical domain, inspired by their great success in the general natural language domain. Among the two main branches of pre-trained language models in the general language domain, i.e., BERT (and its variants) and GPT (and its variants), the first one has been extensively studied in the biomedical domain, such as BioBERT and PubMedBERT. While they have achieved great success on a variety of discriminative downstream biomedical tasks, the lack of generation ability constrains their application scope. In this paper, we propose BioGPT, a domain-specific generative Transformer language model pre-trained on large scale biomedical literature. We evaluate BioGPT on six biomedical NLP tasks and demonstrate that our model outperforms previous models on most tasks. Especially, we get 44.98%, 38.42% and 40.76% F1 score on BC5CDR, KD-DTI and DDI end-to-end relation extraction tasks respectively, and 78.2% accuracy on PubMedQA, creating a new record. Our case study on text generation further demonstrates the advantage of BioGPT on biomedical literature to generate fluent descriptions for biomedical terms. Code is available at https://github.com/microsoft/BioGPT.

keywordsbiomedical literature, generative pre-trained language model, text generation, text mining †

## 1 Introduction

Text mining and knowledge discovery from biomedical literature play important roles in drug discovery, clinical therapy, pathology research, etc. Typical tasks include recognizing named entities in the articles, mining the interaction between drugs and proteins/diseases/other drugs, answering questions given reference text, generating abstracts for given phrases/words, etc. People have accumulated large amounts of literature in the previous studies. For example, PubMed1, one of the most popular biomedical search engines, covers more than 30​M articles and the number still rapidly increases every day as new discoveries are continuously coming out. Therefore, automatically mining the knowledge from literature becomes an urgent demand.

Pre-training models have demonstrated their powerful capability in natural language processing (NLP). On the GLUE benchmark, a widely used benchmark for natural language understanding, pre-training based methods outperform non-pre-training methods by a large margin wang2018glue2. There are two main kinds of pre-training models: (1) the BERT-like models devlin2019bert; liu2019roberta; clark2019electra, mainly for language understanding tasks; (2) the GPT-like models radford2018improving; radford2019language; brown2020language, mainly for language generation tasks.

These models are first pre-trained on large scale corpora collected from the Web via self-supervised learning task (e.g., masked language modeling for BERT, auto-regressive language modeling for GPT), and then fine-tuned on specific donwstream tasks. The BERT-like models are widely used in sequence classification and sequence labeling, where we need to encode the complete document. In comparison, the GPT-like models are often used in generation tasks (e.g., abstract generation, knowledge triplet generation).

By witnessing the success of pre-training in general NLP, people explore adapting these techniques into biomedical domain. However, directly applying these models to the biomedical domain leads to unsatisfactory performance due to domain shift peng-etal-2019-transfer; pubmedbert. A natural solution is to develop pre-training models on biomedical texts (e.g., PubMed). BioBERT biobert and PubMedBERT pubmedbert) are two representative BERT-like models pre-trained on biomedical domain, and they obtain superior performances than general pre-trained models on biomedical benchmarks. However, previous works mainly focus on BERT models which are more appropriate for understanding tasks, not generation tasks. In comparison, GPT models have demonstrated their abilities on generation tasks but demonstrate inferior performance when directly applying to the biomedical domain moradi2021gpt; gutierrez2022thinking.

In this work, we propose BioGPT, a domain-specific generative pre-trained Transformer language model for biomedical text generation and mining. BioGPT follows the Transformer language model backbone, and is pre-trained on 15​M PubMed abstracts from scratch. We apply BioGPT to six biomedical NLP tasks: end-to-end relation extraction on BC5CDR 10.1093/database/baw068, KD-DTI hou2021discovering and DDI herrero2013ddi, question answering on PubMedQA jin2019pubmedqa, document classification on HoC baker2016automatic, and text generation. To adapt to the downstream tasks, we carefully design and analyze the target sequence format and the prompt for better modeling the tasks. Experiments demonstrate that BioGPT achieves better performance compared to baseline methods and other well-performing methods across all the tasks.

## 2 Related Work

### 2.1 Pre-trained Language Models

It has proven to be a very successful pattern in deep learning to pre-train models on large scale unlabeled data via careful designed self-supervision tasks and then transfer to downstream tasks by fine-tuning on them. Downstream tasks can benefit from the learned representations from the pre-trained models. BERT devlin2019bert is a bidirectional transformer based contextualized language model pre-trained on large scale text corpus English Wikipedia and BooksCorpus. It is pre-trained via carefully designed self-supervision tasks: masked language modeling (MLM) task where random word tokens of the input text are replaced by a special token [MASK] which is to be predicted by the model from the context, and the next sentence prediction (NSP) task where two sentences are to be predicted whether the second one is probable given the first one. The pre-trained BERT provides contextualized word representations that can be used by downstream tasks by just fine-tuning on the tasks and has achieved great success on various natural language understanding tasks. Subsequent works mainly focus on pre-training on larger-scale data and models liu2019roberta and advanced pre-training task clark2019electra. Though BERT and various biomedical BERT models have been successful in language understanding tasks and classification tasks, few efforts have been devoted to generative models. As BERT learns powerful word representations through the Transformer encoder model architecture in a bi-directional way, it limits its ability of generation.

Generative Pre-trained Transformer (GPT) radford2018improving is proposed for language generation tasks via pre-training Transformer decoder model on large scale text corpus in a classical casual language modeling task where model learns to predict the next word token only dependent on the previous word tokens. Further, GPT-2 radford2019language and GPT-3 brown2020language with larger model size pre-trained on larger scale text corpus are proposed with remarkable performance in various downstream tasks (e.g., translation, summarization) including classification tasks (e.g., reading comprehension) even without fine-tuning (zero-shot) via appropriate prompts design.

### 2.2 Pre-trained Language Models in Biomedical Domain

When applying to specific domain (e.g., biomedicine), BERT models pre-trained on general domain can be further improved if pre-trained on in-domain text data peng-etal-2019-transfer; beltagy-etal-2019-scibert; biobert. Specifically, biobert and peng-etal-2019-transfer start from the original pre-trained BERT model devlin2019bert that are pre-trained on general domain (Wikipedia and BooksCorpus) and continue pre-training on biomedical literature. Specifically, biobert continue pre-training using PubMed abstracts and PubMed Central full text articles and peng-etal-2019-transfer continue pre-training on both PubMed text and clinical notes from MIMIC-III johnson2016mimic. As they are initialized from the original BERT that are pre-trained on general domain, they use the same vocabulary as the original BERT, which is quite different from the target biomedical domain. Instead of continue pre-training from the pre-trained BERT model, beltagy-etal-2019-scibert pre-train the BERT model from scratch on large corpus of scientific literature (mainly biomedical and computer science literature) where the vocabulary is more suitable for science domain but still contains out-domain information for biomedicine. pubmedbert propose that it is a better strategy to pre-train on domain-specific data from scratch where the vocabulary is more suitable for the biomedical domain. Consequently, they propose PubMedBERT which is pre-trained on 14​M PubMed abstracts from scratch. Similarly, miolo2021electramed pre-train on 28​M data as in peng-etal-2019-transfer also from scratch, using the more advanced ELECTRA model. All these works have shown improved performance on plenty of biomedical literature language processing tasks compared to the original BERT pre-trained on general domain, while none of them is for biomedical generation tasks.

Noticing the powerful generation ability of GPT models, it is quite curious how GPT models perform on biomedical domain which is very different from general domain. However, recent works show that GPT models, even much more powerful GPT-3 model, perform poorly on biomedical tasks moradi2021gpt; gutierrez2022thinking. A previous work on pre-training GPT on biomedical literature is DARE papanikolaou2020dare. However, they pre-train GPT on very limited amount of data (only 0.5​M PubMed abstracts) and use it only for data-augmentation for relation extraction task. A recent work on using GPT model is agrawal2022large, where they design converters for GPT-3 brown2020language for several unconventional downstream clinical tasks.

### 2.3 Downstream Tasks

In this subsection, we introduce the downstream tasks we will work on. A summary of those tasks is in Table 1. All these tasks can be formulated as text generation / mining tasks.

Table 1: Summary of the downstream tasks \toprule Task | Method | Dataset |

\midrule Relation Extraction | GLRE wang2020global, REBEL cabot2021rebel, seq2rel giorgi2022sequence | KD-DTI hou2021discovering, BC5CDR 10.1093/database/baw068, DDI herrero2013ddi |

\midrule Question Answering | QA-Net yu2018qanet, LUKE yamada2020luke, BERT devlin2019bert, PubMedBERT pubmedbert, BioELECTRA kanakarajan-etal-2021-bioelectra, LinkBERT yasunaga2022linkbert | PubMedQA jin2019pubmedqa, BioASQ tsatsaronis2015overview; nentidis2019results |

\midrule Document Classification | BERT devlin2019bert, BlueBERT peng-etal-2019-transfer, SciBERT beltagy-etal-2019-scibert, SPECTER cohan2020specter, PubMedBERT pubmedbert, BioELECTRA kanakarajan-etal-2021-bioelectra, LinkBERT yasunaga2022linkbert | HoC baker2016automatic, SciDocs cohan2020specter |

\botrule | | |

#### 2.3.1 Relation Extraction

Relation extraction is a key task for biomedicine and life science research. Classical pipeline-based methods zeng2014relation; zhou2016attention; wang2020global resolve the task into several separate sub-tasks that require additional intermediate annotations and information which may suffer from the lack of intermediate annotated data and error accumulation. Joint extraction aims to jointly extract the entities and the relations between them from the text. Sequence labeling methods tackle the task by labeling the word tokens in the text with different tags to mark out all the entity mentions and then perform the relation classification between them via classifiers sun2019joint; yuan2020relation; liu2021attention; wei2020novel. Table filling methods formulate the task as a table constituted by the Cartesian product of itself and predicts the relations between the token pairs fu2019graphrel; wang-etal-2020-tplinker; yan-etal-2021-partition. These methods may suffer from error accumulation caused by previous tagging process and laborious intermediate annotations (i.e., named entity recognition). Text generation methods reframe the task as a sequence-to-sequence learning task, by taking the text as the input sequence and the triplet as the target sequence and employing an encoder-decoder network to learn to generate the triplet from the text zeng2018extracting; zhang2020minimize; hou2021discovering; cabot2021rebel; giorgi2022sequence. However, many joint extraction methods still require additional entity information wei2020novel; sui2020joint. In this work, we focus on the end-to-end relation extraction, which formulates the task as an text generation task that takes only the text as the input and generates the relational triplets in an end-to-end way without additional intermediate annotations cabot2021rebel; hou2021discovering; giorgi2022sequence.

#### 2.3.2 Question Answering

Question answering (QA) is the task of answering questions given a context (reading comprehension). Typical methods predict a span in the source context as the answer text, or predicts a label (e.g., yes or no) for simpler tasks with predefined categorical answers yu2018qanet; hu2017reinforced; yamada2020luke. pubmedbert; kanakarajan-etal-2021-bioelectra; yasunaga2022linkbert mainly focus on the biomedical domain question answering task via pre-trained language models. Generative models radford2019language; brown2020language directly generate the answer sequence or the label words.

#### 2.3.3 Document Classification

Document classification is to classify a document into predefined categories (single label or multi label). Recent works on biomedical document classification also leverage large pre-trained language models for understanding the text and predicting the label peng-etal-2019-transfer; pubmedbert; kanakarajan-etal-2021-bioelectra; yasunaga2022linkbert. Generative models radford2019language; brown2020language generate the label words instead of predicting from the predefined set.

## 3 Pre-training Method

In this section, we describe our BioGPT from the perspective of dataset, vocabulary, and model.

Dataset: Dataset is crucial for language model pre-training, in terms of amount, quality and domain. As Gu et al. pubmedbert point, training only on in-domain data from scratch is important for specific domain. Therefore, we only consider in-domain text data and pre-train our model from scratch on the collected data. We collected all the PubMed items3 that were updated before 2021 from the official site4 using the wget tool. We then filtered out all the empty items with only title but no abstract. We used the left 15​M items (each with both title and abstract) as our pre-training dataset.

Vocabulary: pubmedbert also points that in-domain vocabulary is vital. Instead of using the vocabulary of GPT-2, we learn the vocabulary on our collected in-domain corpus. Specifically, we use byte pair encoding (BPE) sennrich2016neural to segment the words in the corpus into word pieces and learn the vocabulary. We adopt the fastBPE5 implementation of BPE. The final learned vocabulary size is 42384.

Model: We adopt the GPT-2 model architecture radford2019language as the backbone of our BioGPT, which is a Transformer decoder vaswani2017attention. Currently we cannot follow the GPT-3 setting due to its extremely large model with 15 billion parameters. The core component of Transformer as well as our BioGPT is the multi-head attention. Given the input, three linear transformations are applied to produce the query Q, the key K and the value V, and then the output is calculated as follows:

| | Multihead​(Q,K,V)=Concat​(h​e​a​d1,h​e​a​d2,⋯,h​e​a​dh)​W, | | (1) |

| | h​e​a​di=softmax​(Qi​KiTd)​Vi, | |

where (1) h is the number of heads; (2) Q, K and V are equally split into Qi, Ki and Vi along the feature dimension, i∈{1,2,⋯,h}; (3) Concat denotes concatenating all inputs as a large tensor along the feature dimension; (4) W is the parameter for the affine transformation. The output of multi-head attention layer is then fed into a feed-forward layer to construct a Transformer layer (or Transformer block). Practically, we adopt GPT-2medium as the backbone network which has 24 layers, 1024 hidden size and 16 attention heads resulting in 355​M parameters in total, and our BioGPT has 347​M parameters (the difference only comes from the different embedding size and output projection size caused by the different vocabulary size).

Training criteria: BioGPT is trained via the standard language modeling task as the same as in radford2018improving; radford2019language. Let 𝒟={xi}i denote the collection of sequences, and sequence xi is made up of ni tokens, i.e., xi=(s1,s2,⋯,sni). The training objective is to minimize the negative log-likelihood:

| min−1|𝒟|​∑i=1|𝒟|∑j=1nilog⁡P⁡(sj|sj−1,sj−2,⋯,s1). | | (2) |

## 4 Fine-tuning Method

In this section, we introduce how to adapt the pre-trained BioGPT to downstream tasks: end-to-end relation extraction, question answering (QA) and document classification. The inputs of the tasks are all sequences, while they have different output formats.

To use BioGPT for these tasks, we need to convert the labels into sequences. In this way, the downstream task is consistent with the pre-training task in terms of the format.

Considering that BioGPT is pre-trained on massive natural language corpus, we convert the labels to sequences in natural language rather than the structured format using special tokens explored in other works hou2021discovering; cabot2021rebel; giorgi2022sequence. In this way, our reformed labels are semantically smoother than using special tokens. We will show the detailed implementation for each task and empirically verify the effectiveness of our method later.

### 4.1 End-to-end Relation Extraction

Task description: Given a source sequence x, we need to find all triplets ⟨head_entityi, tail_entityi, relation⟩Ni=1i, that can be inferred from x. N refers to the number of all possible triplets. Examples include extracting the drug-target-interaction, chemical-disease-relation and drug-drug-interaction.

Method: We convert the triplets into a simple natural language sequence with the same grammatical structures. We explore three forms in this paper:

- 1. the “subject verb object” form (svo), where the entities correspond to the head entity, the relation and the tail entity in the triplet.

- 2. the “subject is the rel.noun of object” form (is-of), where the “rel.noun” refers to the noun form of the relation.

- 3. the “ the relation between subject and object is rel.noun” form (rel-is).

If there are multiple relational triplets for an input document, we sort them according to their order of appearance in the document and use semicolons to concatenated them together.

Let us use a ⟨drug, target, interaction⟩ triplet as example. Suppose we would like to extract triplet ⟨dextropropoxyphene (drug name), mu-type opioid receptor (target name), inhibitor (relation)⟩ from an input document. Then the svo representation is:

dextropropoxyphene inhibits mu-type opioid receptor.

The is-of form is:

dextropropoxyphene is the inhibitor of mu-type opioid receptor.

The rel-is form is:

the relation between dextropropoxyphene

and mu-type opioid receptor is inhibitor.

The natural sentences can be converted back to triplets using regular expression. Users can also design customized formats depending on tasks.

### 4.2 Question Answering

Task description Given a question, a reference context and an answer, the goal is to answer the question given the reference context. The label is within the category of yes, no, or maybe.

Method: We pre-pend the description word “question:” and “context:” before the question and the context respectively and concatenate them together as the source sequence. Then for the target sequence, we generate it using the format “the answer to the question given the context is label”. For example:

source: question: question text. context: context text. target: the answer to the question given the context is yes.

### 4.3 Document Classification

Task description Given a document text, the goal is to classify the type of the document.

Method: We generate the target sequence using the format “the type of this document is label”. For example:

the type of this document is genomic instability and mutation.

### 4.4 Prompt-based Fine-tuning

Figure 1: Framework of BioGPT when adapting to downstream tasks We have formatted the labels to target sequences. The last question is, how do we use the source and the target to fine-tune and inference with BioGPT? A naive way is to concatenate the source and the target sequences together but is difficult for the model to generate during inference as it does not know what to generate for the specific task given the source text input.

Prompt is recently extensively explored in NLP liu2021pre to elicit the knowledge from a pre-trained language model. Prompt is to append task-specific instructions to the input for the model to better generate output that meets the demand of the task. GPT-3 brown2020language uses hard prompts (manually designed discrete language phrases) to generate for different tasks. Though hard prompts can achieve satisfactory performance, designing task specific prompts is laborious and it is found that different prompts lead to different performance.

In this work, we mainly adopt soft prompts in prefix-tuning li-liang-2021-prefix, which leverage continuous embeddings (virtual tokens) to steer the pre-trained language model by directly appending several additional virtual tokens before the text as the prompts. Such continuous embeddings are randomly initialized and learned end-to-end on the downstream tasks to be task-specific. Different from li-liang-2021-prefix, we do not append the virtual tokens to the very beginning of the source input, but only before the target sequence (between the source and the target). Equipped with the prompt, our final sequence is constructed as [source;prompt;target], as depicted in Fig. 1. During the inference, we provide the source text and the prompt as the prefix for the language model to condition on and let the language model to generate the target output as in Fig. 1.

## 5 Experiments

In this section, we pre-train our BioGPT and evaluate it on the following four biomedical NLP tasks across six datasets: end-to-end relation extraction on BC5CDR 10.1093/database/baw068, KD-DTI hou2021discovering and DDI herrero2013ddi, question answering on PubMedQA jin2019pubmedqa, document classification on HOC baker2016automatic, and text generation on self-created dataset. We use fairseq ott2019fairseq as our code base for implementation. We adopt the GPT-2medium model configuration as our backbone model configuration. We perform BPE to learn to tokenize the corpus and construct the vocabulary instead of using the learned vocabulary from GPT-2 due to the domain gap between the biomedical domain and the general domain.

For pre-training, we pre-train BioGPT on 8 NVIDIA V100 GPUs for 200​k steps, with 1024 tokens per GPU and 64 accumulated steps (i.e., the final batch size is 1024×8×64=524288 tokens). We use Adam kingma2015adam as the optimizer with a peak learning rate of 2×10−4 and 20000 warm-up steps. The learning rate follows an inverse square root decay schedule after reaching the peak as in vaswani2017attention.

All the fine-tuning experiments are conducted on a single NVIDIA V100 GPU, with a batch size of 1024 tokens and 32 accumulated steps.

During the inference, we adopt beam search with beam size=5 for the text generation task, and greedy search for all the other tasks.

We make comparison to general domain GPT-2 for all the experiments. Specifically, we use the GPT-2medium model from the Hugging face library wolf-etal-2020-transformers6 which is the backbone network of our BioGPT.

### 5.1 End-to-end Relation Extraction

Relation extraction is an important task in information extraction. Here we target at the end-to-end relation extraction setting where the model takes the text as the input and directly generates the relational triplets. We mainly compare to REBEL cabot2021rebel, a recently proposed end-to-end triplet extraction approach based on sequence-to-sequence model, which employs BART pre-trained model lewis-etal-2020-bart as the backbone model, and further enhances it by pre-training on additional large relational triplet dataset created from Wikipedia as REBELpt.

#### 5.1.1 BC5CDR

BC5CDR is a dataset for chemical-disease-relation extraction task introduced by 10.1093/database/baw068 which consists of 500/500/500 documents as the training/validation/test set. We fine-tune GPT-2medium and BioGPT for 100 epochs with a peak learning rate 10−5 and 100 warm-up steps. We use continuous embeddings with length=9 as prompts and the rel-is target sequence format. Since BC5CDR is a binary relation dataset where the entities are labeled if the relationship exists instead of a specific relation type, we use the pattern “the relation between head_entity and tail_entity exists” as the target sequence format. We average the checkpoints of the last 5 epochs for evaluation. We mainly measure and compare the micro-F1 score. We compare BioGPT to REBEL and seq2rel giorgi2022sequence where both methods are end-to-end relation extraction methods based on sequence-to-sequence modeling. We also compare with a pipeline-based extraction method, GLRE wang2020global which requires NER (named entity recognition) information as the intermediate annotations in the pipeline. Originally, GLRE uses the ground truth NER information. To make a fair comparison, we experiment with GLRE for two settings: 1) using ground-truth NER information during the training and using open-source NER tool during the inference (i.e., GLRE (gt+pred)) and 2) using open-source NER tool for both the training and the inference (i.e., GLRE (pred+pred)). We use the open-source NER tool7 for the NER tagging. We try our best to run the baseline methods and evaluate them.

Table 2: Results on BC5CDR chemical-disease-relation extraction task. ’gt+pred’ means using ground truth NER information for training and using open-source NER tool to annotate NER for inference. ’pred+pred’ means using open-source NER tool for both training and inference. ’†’ means training on training and validation set. \topruleModel | Precision | Recall | F1 |

\midruleGLRE (gt+pred) | 34.82 | 18.29 | 23.99 |

GLRE (pred+pred) | 23.00 | 4.88 | 8.05 |

\midruleGPT-2 radford2019language | 43.92 | 32.55 | 37.39 |

REBEL cabot2021rebel | 34.28 | 39.49 | 36.70 |

REBELpt cabot2021rebel | 40.94 | 21.20 | 27.94 |

seq2rel giorgi2022sequence† | 43.5 | 37.5 | 40.2 |

\midruleBioGPT | 49.44 | 41.28 | 44.98 |

BioGPT† | 49.52 | 43.25 | 46.17 |

\botrule | | | |

From the results in Table 2, we can see that BioGPT achieves the best result (44.98%) among all the methods, with large improvements. We have several findings: 1) pipeline-based method GLRE significantly drops when using NER tagged by open-source tools instead of ground truth NER. However, this is often the common case in practical situation where the annotations for NER are lacked or expensive to collect. When applying open-source NER tools to some specific domains, errors occur and lead to inferior performance of relation extraction. 2) Compared to REBEL, BioGPT has a large gain with 8.28% improvement. Notice that seq2rel giorgi2022sequence is trained on both the training set and validation set, while our BioGPT is only trained on the training set and still outperforms it with 4.78% improvement. Moreover, when also trained on both the training set and the validation set, BioGPT further improves to 46.17% with 5.97% improvement against seq2rel giorgi2022sequence.

#### 5.1.2 KD-DTI

KD-DTI is dataset for drug-target-interaction introduced by hou2021discovering, consisting of 12​k/1​k/1.3​k documents as the train/validation/test set. We fine-tune GPT-2medium and BioGPT on the task for 30 epochs using Adam optimizer with a peak learning rate of 10−5 and 1000 warm-up steps. We use continuous embeddings with length=9 as prompts and the rel-is target sequence format for constructing the target sequence. We average the checkpoints of the last 5 epochs for evaluation. We mainly measure and compare the micro-F1 score and the results are listed in Table 3.

Table 3: Results on KD-DTI drug-target-interaction extraction task \topruleModel | Precision | Recall | F1 |

\midruleTransformer + PubMedBERT | 25.35 | 24.14 | 24.19 |

-attn hou2021discovering |

GPT-2medium | 30.53 | 27.87 | 28.45 |

REBEL | 32.36 | 29.58 | 30.39 |

REBELpt | 35.73 | 32.61 | 33.32 |

\midruleBioGPT | 40.00 | 39.72 | 38.42 |

\botrule | | | |

We compare BioGPT with GPT-2medium, Transformer + PubMedBERT-attn evaluated in hou2021discovering and REBEL. It can be shown that BioGPT achieves 38.42% f1 score, with 14.23%, 9.97% and 8.03% improvement compared to Transformer + PubMedBERT-attn, GPT-2medium and REBEL. Particularly, it surpasses REBELpt by 5.1% which is further pre-trained on large relation extraction dataset while BioGPT does not.

#### 5.1.3 DDI

DDI extraction 2013 corpus is a dataset for drug-drug-interaction task introduced by herrero2013ddi, consisting of 792 texts selected from the DrugBank database and other 233 Medline abstracts. We use the original dataset and use a train/validation/test split of 664/50/191 files. We fine-tune GPT-2medium and BioGPT for 100 epochs with a peak learning rate 10−4 and 500 warm-up steps. We also use continuous embeddings with length=9 as prompts and the rel-is target sequence format. The last 5 epochs are averaged for evaluation. The micro-F1 score is measured and compared.

Table 4: Results on DDI drug-drug-interaction extraction task \topruleModel | Precision | Recall | F1 |

\midruleGPT-2medium | 23.39 | 31.93 | 24.68 |

REBEL | 35.36 | 28.64 | 28.27 |

REBELpt | 46.59 | 39.60 | 40.56 |

\midruleBioGPT | 41.70 | 44.75 | 40.76 |

\botrule | | | |

The results are shown in Table 4 from which we can see that BioGPT achieves 40.76% with 16.08% and 12.49% improvement against GPT-2medium and REBEL. It also surpasses REBELpt which uses additional large relation extraction dataset for two-stage pre-training.

### 5.2 Question Answering

PubMedQA jin2019pubmedqa is a biomedical question answering dataset. Each sample is constructed from a PubMed abstract, containing a question, a reference context, a long answer, and a yes/no/maybe label which is the answer to the question. We use the original train/validation/test split with 450, 50 and 500 respectively, noted as PQA-L in jin2019pubmedqa for evaluation. We also use the additional dataset noted as PQA-A and PQA-U in jin2019pubmedqa for fine-tuning. We use the continuous embedding with length=9 as the soft prompt. We format the data into source sequence and target sequence as described before. We apply techniques such as two-stage fine-tuning jin2019pubmedqa and noisy labels to improve the performance. We measure and compare the classification accuracy of the reasoning required setting described in jin2019pubmedqa.

Table 5: Results on PubMedQA question answering task \topruleModel | Accuracy |

\midrulePubMedBERT pubmedbert | 55.8 |

BioELECTRa kanakarajan-etal-2021-bioelectra | 64.2 |

BioLinkBERTbase yasunaga2022linkbert | 70.2 |

BioLinkBERTlarge yasunaga2022linkbert | 72.2 |

\midruleBioGPT | 78.2 |

\botrule | |

From the results in Table 5 we can see that BioGPT achieves 78.2% accuracy with 6.0% improvement over previous best performance obtained by BioLinkBERT yasunaga2022linkbert, achieving a new state-of-the-art on this task.

### 5.3 Document Classification

HoC (the Hallmarks of Cancers corpus) consists of 1580 PubMed abstracts manually annotated at sentence level by experts with ten currently known hallmarks of cancer baker2016automatic. We follow the same training/test split as in peng-etal-2019-transfer. We use the continuous embedding with length=1 as the prompt and format the label into the target sequence as described before. We fine-tune GPT-2medium and BioGPT for 20000 steps with a peak learning rate 10−5 and 1000 warm-up steps. Micro-F1 score is measured and reported for comparison.

Table 6: Results on HoC document classification task \topruleModel | F1 |

\midruleBioBERT biobert | 81.54 |

PubMedBERT pubmedbert | 82.32 |

PubMedBERTlarge | 82.70 |

BioLinkBERTbase | 84.35 |

GPT-2medium | 81.84 |

\midruleBioGPT | 85.12 |

\botrule | |

We can see from the results in Table 6 that BioGPT achieves 85.12% accuracy with 3.28% improvement over general domain GPT-2, and surpasses BioBERT, PubMedBERT and BioLinkBERT with 3.58%, 2.8%, 0.77% improvements respectively.

### 5.4 Text Generation

GPT, GPT-2 and GPT-3 demonstrate remarkable text generation ability. Given words, phrases or simple sentences as prefix, they can continue to generate text that are syntactically correct and semantically smooth conditioning on the given text. We are also curious about the text generation ability of the pre-trained BioGPT in the biomedical domain, and how does general domain GPT-2 perform in the biomedical domain.

We evaluate the biomedical text generation ability of BioGPT and GPT-2medium. Specially, we extract all the entities within the triplets from the KD-DTI test set (i.e., drugs and targets). Then for each drug/target name, we provide it to the language model as the prefix and let the model generate text conditioned on it. We then investigate whether the generated text is meaningful and fluent.

For this task, no objective evaluation metric is reported here. Instead, we provide a few examples here for demonstration.

Table 7: Example of generated text \toprule Input | Model | Text |

\midrule Bicalutamide | GPT-2 | Bicalutamide, a synthetic bicalutamide derivative, was found to increase the cellular proliferation of C. elegans cells. This effect was not observed with the other synthetic bicalutamide derivatives. This is in line with our previous work which showed that bicalutamide increases the proliferation of C. elegans cells but not in the presence of synthetic bicalutamide. |

| BioGPT | Bicalutamide (Casodex) is an androgen receptor (AR) antagonist approved for the treatment of metastatic castration-resistant prostate cancer (mCRPC) in patients who have progressed on or are ineligible for docetaxel chemotherapy, as well as for the treatment of early-stage prostate cancer in men who have not progressed on or are ineligible for docetaxel chemotherapy, as well as for the treatment of metastatic castration-sensitive prostate cancer (mCSPC) in men who have not progressed on or are ineligible for docetaxel chemotherapy. |

\midrule Janus kinase 3 (JAK-3) | GPT-2 | Janus kinase 3 (JAK-3) mediates the conversion of glucose to glycogen in skeletal muscle. The increase in glucose uptake by skeletal muscle is believed to be associated with a reduction in muscle protein breakdown and with the release of ketone bodies, which contribute to the metabolism of glucose. |

| BioGPT | Janus kinase 3 (JAK-3) is a member of the Janus kinase (JAK) family of non-receptor tyrosine kinases and plays an important role in the regulation of cell proliferation, differentiation, survival, migration, and angiogenesis. |

\midrule Apricitabine | GPT-2 | Apricitabine, duloxetine, and citalopram) were investigated in patients with a history of treatment-resistant depression (n = 10). We found that treatment with the three drugs significantly decreased depressive symptoms, but not depressive symptoms over the next 12 weeks. We conclude that antidepressant drugs that reduce depressive symptoms and reduce depressive symptoms over time may be beneficial in patients with recurrent depression. |

| BioGPT | Apricitabine is an oral prodrug of 5-aza-2 ’-deoxycytidine (5-aza-CdR), a DNA methyltransferase (DNMT) inhibitor, which has been approved by the US Food and Drug Administration (FDA) for the treatment of myelodysplastic syndrome (MDS) and acute myeloid leukemia (AML) in combination with low-dose cytarabine (Ara-C) and granulocyte colony-stimulating factor (G-CSF) for patients with intermediate-2 or high-risk MDS or AML. |

\midrule Xylazine | GPT-2 | Xylazine, the ”active ingredient” in ”bath salts” (see above) has been linked to numerous deaths, including those of former University of California, Berkeley, professor Andrew Anglin, who posted a video of himself having a bath filled with it. |

| BioGPT | Xylazine is an alpha 2-adrenoceptor agonist that has been used as a sedative and analgesic in veterinary medicine for many years, but its effects on the cardiovascular system have not been extensively studied in the dog, and its effects on the central nervous system (CNS) have not been well characterized in the dog, despite the fact that xylazine has been widely used as a sedative and analgesic in veterinary medicine for more than 30 years. |

\midrule Psoralen | GPT-2 | Psoralen, the city of the spiritualist, and had been given to him by God, who wished to show him the way of salvation, and to draw him back to the true life of Christ, and to bring him into communion with his fellow-man. |

| BioGPT | Psoralen photochemotherapy (PUVA) is a well-established treatment for psoriasis, but its use is limited by the risk of skin cancer, particularly squamous cell carcinoma (SCC) of the head and neck (H & N), which is the most common site of skin cancer in the United States (US) |

\midrule CP-673451 | GPT-2 | CP-673451) and 4-5-6-3-1-2-3-4-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29-30-31-32-33-34-35-36-37-38-39-40-41-42-43-44-45-46-47-48-49-50… |

| BioGPT | CP-673451 is a potent, selective, and orally active inhibitor of human neutrophil elastase (HNE) and human cathepsin G (CatG) with in vitro and in vivo anti-inflammatory activity in a variety of animal models of inflammation and in a model of acute lung injury (ALI) in the rat induced by intratracheal instillation of lipopolysaccharide (LPS) and tumor necrosis factor-alpha (TNF-alpha), a model of acute lung injury (ALI) in which neutrophils play an important role. |

\midrule BIIB-021 | GPT-2 | BIIB-021A-2-1.pdf. Wright, C. (2002). ”The War on Terror: A Legal Theory of Terrorist Attacks,” in L.E.R. Wright, S.R. Warrick, and J.H. Rothman (Eds.), The War on Terror: Essays in Honor of Michael T. Klare (New York: Oxford University Press), 143-57. |

| BioGPT | BIIB-021 is a novel, orally active, non-peptide bradykinin B2 receptor antagonist with potent and long-lasting anti-inflammatory activity in animal models of acute and chronic inflammation and in a rat model of adjuvant-induced arthritis (AIA), an animal model of rheumatoid arthritis (RA), and in a rat model of collagen-induced arthritis (CIA), an animal model of collagen-induced arthritis (CIA), in which arthritis is induced by immunization with bovine type II collagen (CII). |

\botrule | | |

Table 8: Generated text of keywords of interest \toprule Input | Model | Text |

\midrule COVID-19 | GPT-2 | COVID-19-0067-7 Hip-hop-romance-detective-franchise-unveiled-in-coco-houston http://www.pinterest.com/pin/71178278499149422/ Black Widow: World War II’s Greatest Widow? http://www.pinterest.com/pin/71177842668650112/ |

| BioGPT | COVID-19 is a global pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the causative agent of coronavirus disease 2019 (COVID-19), which has spread to more than 200 countries and territories, including the United States (US), Canada, Australia, New Zealand, the United Kingdom (UK), and the United States of America (USA), as of March 11, 2020, with more than 800,000 confirmed cases and more than 800,000 deaths. |

\midrule SARS-CoV-2 | GPT-2 | SARS-CoV-2-CKM-T1-1/pdf/T1/SARS-CoV-2-CKM-T1-1.pdf Cancer http://www.cancer.gov/cancer topics/programs/tox/tox-programs.html Other Research http://www.cdc.gov/cancer/cancer/index.html |

| BioGPT | SARS-CoV-2 is the causative agent of COVID-19, a severe acute respiratory syndrome (SARS) that has infected more than 390,000 people worldwide and killed more than 250,000 people. |

\midrule The treatment of COVID-19 | GPT-2 | The treatment of COVID-19 is similar to that of COVID-20 and COVID-22, in which COVID-19 is treated with a variety of ionic, dopaminergic and serotonergic agents. In addition to its effects on COVID-19, the use of COVID-19 has been suggested to exert effects on the function of other dopamine neurons, including dopamine D2 receptors, D3 receptors and serotonin transporters. These effects include attenuation of the increase in dopamine release induced by the central nervous system stimulant drug amphetamine and its metabolites. |

| BioGPT | The treatment of COVID-19 with remdesivir (remdesivir for COVID-19) has been approved by the US Food and Drug Administration (FDA) for the treatment of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) infection in patients with confirmed SARS-CoV-2 infection and is currently being evaluated in clinical trials for the treatment of COVID-19 in patients with confirmed SARS-CoV-2 infection, as well as in patients with suspected or confirmed SARS-CoV-2 infection. |

\midrule The drug that can treat COVID-19 is | GPT-2 | The drug that can treat COVID-19 is not effective enough to cure patients with this rare genetic disease, but it could help patients who have it. Researchers say this finding could mean that a drug could be developed that would help patients with this rare genetic disorder. The study, published in the journal Molecular Psychiatry, also looked at how a drug might be used to treat patients with COPD, which causes chronic pain and lung damage. Researchers say the study could also lead to new drugs that treat COPD. |

| BioGPT | The drug that can treat COVID-19 is hydroxychloroquine (HCQ), which has been shown to inhibit the replication of the SARS-CoV-2 virus in cell culture and in animal models of COVID-19, and has been approved by the US Food and Drug Administration (FDA) for the treatment of patients with COVID-19 in March 2020, and by the European Medicines Agency (EMA) for the treatment of patients with COVID-19 in April 2020, and by the European Medicines Agency (EMA) for the treatment of patients with COVID-19 in May 2020. |

\midrule Omicron variants of SARS-CoV-2 | GPT-2 | Omicron variants of SARS-CoV-2 have been detected in the blood of patients who contracted SARS-CoV-2 and survived it, but it is unclear whether these variants were caused by SARS-CoV-2 infection or if the SARS-CoV-2 variant in the blood was due to a misidentification of SARS-CoV-2 from other SARS strains. We found that a polymorphism in the gene encoding the SARS-CoV-2-specific viral surface protein was associated with SARS-CoV-2 infection in a cohort of patients with SARS-CoV-2 infection who had an active SARS infection, suggesting that SARS-CoV-2 may be able to infect the host during an active infection. |

| BioGPT | Omicron variants of SARS-CoV-2 have been isolated from patients with severe acute respiratory syndrome (SARS) and have been shown to be highly pathogenic in mice and ferrets, suggesting that they may play a role in the pathogenesis of SARS-CoV-2 infection and the development of severe disease in patients with SARS-CoV-2 infection. |

\botrule | | |

From the results in Table 7, we can see that: (1) Given relatively common names as input, for example in the first two cases (i.e., Bicalutamide and JAK-3), GPT-2 can generate meaningful and fluent text that is related to the word and biomedicine, while BioGPT generates more specific and professional descriptions. (2) When given some uncommon names (e.g., in the Apricitabine and Xylazine cases), GPT-2 cannot generate meaningful descriptions while BioGPT still generates specific descriptions. Especially in the Apricitabine case, GPT-2 seems to generate a piece of text that comes from a specific scientific paper while BioGPT generates more general description. (3) When given some very uncommon and domain specific names that even lose semantic information from their surface names (e.g., Psoralen, CP-673451 and BIIB-021), GPT-2 trained on general completely failed to generate any informative text. Given Psoralen, GPT-2 treats it as a city name and generates some text though fluent but unrelated to the given name. Given CP-673451, GPT-2 even begins to count numbers. Given BIIB-021, GPT-2 treats it as a name of a pdf document. For these types, BioGPT is still able to generate text that describes the names or is highly related to them.

Besides these samples, we also manually input several keywords or phrases that are of interest (e.g., COVID-19 related terms) and see what GPT-2 and our BioGPT generate. The results are listed in Table 8, where we input many COVID-19 related key words/phrases as the prefix for the language model to condition on. We can see that GPT-2 treats the term ‘‘COVID-19’’ and ‘‘SARS-CoV-2’’ as some codes within a link or file name rather the entities we care about while BioGPT can generate clear descriptions. More interestingly, when prompting ‘‘The drug that can treat COVID-19 is’’, BioGPT is able to answer it with the drug ‘‘hydroxychloroquine’’ which is indeed noticed at MedlinePlus8. Notice that GPT-2 is pre-trained on the corpus before COVID-19 while BioGPT is pre-trained on the corpus before 2021 that contains COVID-19 information, therefore it is not surprising that BioGPT performs much better than GPT-2 on COIVD-19 related key words in Table 8. However, in the last example in Table 8, both models do not have any knowledge of the Omicron variants of SARS-CoV-2 which appear in the late 2021, while BioGPT still generates more fluent and relevant text compared to GPT-2.

Overall, we can see that BioGPT pre-trained on in-domain biomedical literature from scratch performs better than general domain GPT-2 across various biomedical NLP tasks, and performs better than most previous methods on respective tasks, achieving state-of-the-art on four out of six tasks.

## 6 Ablation Study

In this section, we conduct ablation study on the prompt design and the target sequence format of the label.

Table 9: Results on KD-DTI with different target formats \topruleTarget format | Precision | Recall | F1 |

\midrule<head> head_entity <tail> tail_entity <relation> relation | 38.21 | 40.21 | 37.32 |

svo (head_entity relation tail_entity) | 37.95 | 37.77 | 36.57 |

is-of (head_entity is the relation of tail_entity) | 39.37 | 39.11 | 37.77 |

rel-is (the relation between head_entity and tail_entity is relation) | 38.93 | 40.70 | 38.38 |

\botrule | | | |

### 6.1 Target Sequence Format

Previous works cabot2021rebel; phan2021scifive; giorgi2022sequence; hou2021discovering directly format the labels into structured formats using special tokens. Taking the triplet generation task as an example, in REBEL cabot2021rebel, the triplets are represented by:

<triplet> head_entity1 <subj> tail_entity1 <obj> relation1 <triplet> head_entity2 <subj> tail_entity2 <obj> relation2 ⋯,

where <triplet>, <subj> and <obj> are special tokens to represent the start of the head entity, the tail entity and the relation. cabot2021rebel; hou2021discovering; giorgi2022sequence use similar method to process the targets.

Although these methods achieved promising results in their tasks respectively, such formulation pattern is not the best choice for BioGPT. Previous works use an encoder-decoder framework, where two separated modules are leveraged to process the input (by the encoder) and generate the answers (by the decoder). The two modules can be trained to fit the two different types of sequences (natural language sequence v.s. structured sequence).

In contrast, in BioGPT, we use a unified module to encode context and generate answers. Intuitively, it is better to maintain the format consistency between the inputs and answers. Consequently, instead of the structured target format with special tokens as in previous works, we format the label within a natural language sentence for the language model to smoothly learn and generate. However, there are also various patterns that can be used to construct the target sentence. We explore several target sequence formats, including the structured format, on the KD-DTI dataset for end-to-end relation extraction task. We fix the prompt to continuous embeddings with length=9. From the results in Table 9 we can see that the formats in natural language perform better than structured format, and that the rel-is format performs the best among all the formats in terms of F1 which provides a more semantically smooth and clear description. We also conduct experiments on BC5CDR and DDI to further compare the structure format and the rel-is format. The F1 scores of the structure format on BC5CDR and DDI are 42.85 and 38.60, while those two scores with rel-is format are 44.98 and 40.76, which further verify our conclusion.

### 6.2 Prompt Design

Table 10: Results on KD-DTI with different prompts \toprulePrompts | Precision | Recall | F1 |

\midrulewe have that | 38.55 | 38.37 | 36.95 |

in conclusion, | 39.03 | 39.45 | 37.76 |

we can conclude that | 39.56 | 39.88 | 38.16 |

continuous embeddings (length=1) | 39.50 | 39.71 | 38.06 |

continuous embeddings (length=5) | 39.57 | 39.63 | 38.09 |

continuous embeddings (length=9) | 38.93 | 40.70 | 38.38 |

continuous embeddings (length=13) | 39.48 | 39.17 | 38.60 |

continuous embeddings (length=17) | 39.82 | 39.60 | 38.28 |

\botrule | | | |

We conduct experiment with manually designed hard prompts and continuous embedding soft prompts on the KD-DTI extraction task. We fix the target format to the rel-is format (i.e., ”the relation between head_entity and tail_entity is relation”). From the results in Table 10 we can see that the best performing prompt is continuous embeddings with length of 13 virtual tokens. Moreover, we have several observations: (1) Different manually designed hard prompts result in different performance and more instructive and informative prompt (e.g., “we can conclude that”) achieve better performance. (2) Generally, continuous embedding soft prompts perform better than manually designed hard prompts. (3) The performance of the continuous embedding soft prompts are roughly irrelevant to the length. In our previous experiments, we empirically choose length=9 according to the performance on validation set.

## 7 Conclusion

In this work, we proposed BioGPT, a generative pre-trained Transformer language model for biomedical text generation and mining. We adopted GPT-2 as our backbone model and pre-trained on 15​M PubMed abstracts corpus. We carefully designed and investigated the prompt and the target sequence format when applying pre-trained BioGPT to downstream tasks. We applied the pre-trained BioGPT to biomedical NLP tasks: end-to-end relation extraction task, question answering task, document classification task and text generation task. BioGPT achieves SOTA results on three end-to-end relation extraction tasks and one question answering task. It also demonstrates better biomedical text generation ability compared to GPT-2 on the text generation task. For future work, we plan to train larger scale BioGPT on larger scale biomedical data and apply to more downstream tasks.

Key Points Our contributions are summarized as follows: • We propose BioGPT, a generative pre-trained Transformer language model on biomedical domain. BioGPT can be used for biomedical literature text generation and mining. • BioGPT achieves state-of-the-art results on four benchmarks: BC5CDR, KD-DTI and DDI end-to-end relation extraction task, and PubMedQA question answering task. We also demonstrate the capability of biomedical text generation of BioGPT compared to standard GPT trained on general domain. • We study the prompt design and the target sequence design when applying BioGPT to downstream tasks and find that target sequence with natural language semantics are better than structured prompts explored in previous works.

{appendices}

## 8 Scaling to Larger Size

We also scaled our model to larger size. We built BioGPT-Large, based on the GPT-2 XL architecture (the largest version of GPT-2), with 1.5B model parameters. We fine-tune and evaluate its performance on the downstream tasks, as shown in Table 11.

Table 11: Performance of BioGPT-Large fine-tuned on downstream tasks \topruleTask | Metric | Performance |

\midruleBC5CDR | F1 | 50.12 |

KD-DTI | F1 | 38.39 |

DDI | F1 | 44.89 |

PubMedQA | Accuracy | 81.0 |

HoC | F1 | 84.40 |

\botrule | | |

## References

- [1] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2019.

- [2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, 2019.

- [3] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

- [4] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. Electra: Pre-training text encoders as discriminators rather than generators. In International Conference on Learning Representations, 2019.

- [5] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.

- [6] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

- [7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- [8] Yifan Peng, Shankai Yan, and Zhiyong Lu. Transfer learning in biomedical natural language processing: An evaluation of BERT and ELMo on ten benchmarking datasets. In Proceedings of the 18th BioNLP Workshop and Shared Task, pages 58–65, Florence, Italy, August 2019. Association for Computational Linguistics.

- [9] Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare (HEALTH), 3(1):1–23, 2021.

- [10] Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4):1234–1240, 09 2019.

- [11] Milad Moradi, Kathrin Blagec, Florian Haberl, and Matthias Samwald. Gpt-3 models are poor few-shot learners in the biomedical domain. arXiv preprint arXiv:2109.02555, 2021.

- [12] Bernal Jiménez Gutiérrez, Nikolas McNeal, Clay Washington, You Chen, Lang Li, Huan Sun, and Yu Su. Thinking about gpt-3 in-context learning for biomedical ie? think again. arXiv preprint arXiv:2203.08410, 2022.

- [13] Jiao Li, Yueping Sun, Robin J. Johnson, Daniela Sciaky, Chih-Hsuan Wei, Robert Leaman, Allan Peter Davis, Carolyn J. Mattingly, Thomas C. Wiegers, and Zhiyong Lu. BioCreative V CDR task corpus: a resource for chemical disease relation extraction. Database, 2016, 05 2016.

- [14] Yutai Hou, Yingce Xia, Lijun Wu, Shufang Xie, Yang Fan, Jinhua Zhu, Wanxiang Che, Tao Qin, and Tie-Yan Liu. Discovering drug-target interaction knowledge from biomedical literature. arXiv preprint arXiv:2109.13187, 2021.

- [15] María Herrero-Zazo, Isabel Segura-Bedmar, Paloma Martínez, and Thierry Declerck. The ddi corpus: An annotated corpus with pharmacological substances and drug–drug interactions. Journal of biomedical informatics, 46(5):914–920, 2013.

- [16] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2567–2577, 2019.

- [17] Simon Baker, Ilona Silins, Yufan Guo, Imran Ali, Johan Högberg, Ulla Stenius, and Anna Korhonen. Automatic semantic classification of scientific literature according to the hallmarks of cancer. Bioinformatics, 32(3):432–440, 2016.

- [18] Iz Beltagy, Kyle Lo, and Arman Cohan. SciBERT: A pretrained language model for scientific text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3615–3620, Hong Kong, China, November 2019. Association for Computational Linguistics.

- [19] Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. Scientific data, 3(1):1–9, 2016.

- [20] Giacomo Miolo, Giulio Mantoan, and Carlotta Orsenigo. Electramed: a new pre-trained language representation model for biomedical nlp. arXiv preprint arXiv:2104.09585, 2021.

- [21] Yannis Papanikolaou and Andrea Pierleoni. Dare: Data augmented relation extraction with gpt-2. arXiv preprint arXiv:2004.13845, 2020.

- [22] Monica Agrawal, Stefan Hegselmann, Hunter Lang, Yoon Kim, and David Sontag. Large language models are zero-shot clinical information extractors. arXiv preprint arXiv:2205.12689, 2022.

- [23] Difeng Wang, Wei Hu, Ermei Cao, and Weijian Sun. Global-to-local neural networks for document-level relation extraction. arXiv preprint arXiv:2009.10359, 2020.

- [24] Pere-Lluís Huguet Cabot and Roberto Navigli. Rebel: Relation extraction by end-to-end language generation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2370–2381, 2021.

- [25] John Giorgi, Gary D Bader, and Bo Wang. A sequence-to-sequence approach for document-level relation extraction. arXiv preprint arXiv:2204.01098, 2022.

- [26] Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi, and Quoc V Le. Qanet: Combining local convolution with global self-attention for reading comprehension. arXiv preprint arXiv:1804.09541, 2018.

- [27] Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, and Yuji Matsumoto. Luke: deep contextualized entity representations with entity-aware self-attention. arXiv preprint arXiv:2010.01057, 2020.

- [28] Kamal raj Kanakarajan, Bhuvana Kundumani, and Malaikannan Sankarasubbu. BioELECTRA:pretrained biomedical text encoder using discriminators. In Proceedings of the 20th Workshop on Biomedical Language Processing, pages 143–154, Online, June 2021. Association for Computational Linguistics.

- [29] Michihiro Yasunaga, Jure Leskovec, and Percy Liang. Linkbert: Pretraining language models with document links. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8003–8016, 2022.

- [30] George Tsatsaronis, Georgios Balikas, Prodromos Malakasiotis, Ioannis Partalas, Matthias Zschunke, Michael R Alvers, Dirk Weissenborn, Anastasia Krithara, Sergios Petridis, Dimitris Polychronopoulos, et al. An overview of the bioasq large-scale biomedical semantic indexing and question answering competition. BMC bioinformatics, 16(1):1–28, 2015.

- [31] Anastasios Nentidis, Konstantinos Bougiatiotis, Anastasia Krithara, and Georgios Paliouras. Results of the seventh edition of the bioasq challenge. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 553–568. Springer, 2019.

- [32] Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S Weld. Specter: Document-level representation learning using citation-informed transformers. arXiv preprint arXiv:2004.07180, 2020.

- [33] Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. Relation classification via convolutional deep neural network. In Proceedings of COLING 2014, the 25th international conference on computational linguistics: technical papers, pages 2335–2344, 2014.

- [34] Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, and Bo Xu. Attention-based bidirectional long short-term memory networks for relation classification. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 207–212, Berlin, Germany, August 2016. Association for Computational Linguistics.

- [35] Changzhi Sun, Yeyun Gong, Yuanbin Wu, Ming Gong, Daxin Jiang, Man Lan, Shiliang Sun, and Nan Duan. Joint type inference on entities and relations via graph convolutional networks. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1361–1370, 2019.

- [36] Yue Yuan, Xiaofei Zhou, Shirui Pan, Qiannan Zhu, Zeliang Song, and Li Guo. A relation-specific attention network for joint entity and relation extraction. In IJCAI, volume 2020, pages 4054–4060, 2020.

- [37] Jie Liu, Shaowei Chen, Bingquan Wang, Jiaxin Zhang, Na Li, and Tong Xu. Attention as relation: learning supervised multi-head self-attention for relation extraction. In Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence, pages 3787–3793, 2021.

- [38] Zhepei Wei, Jianlin Su, Yue Wang, Yuan Tian, and Yi Chang. A novel cascade binary tagging framework for relational triple extraction. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 1476–1488, 2020.

- [39] Tsu-Jui Fu, Peng-Hsuan Li, and Wei-Yun Ma. Graphrel: Modeling text as relational graphs for joint entity and relation extraction. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1409–1418, 2019.

- [40] Yucheng Wang, Bowen Yu, Yueyang Zhang, Tingwen Liu, Hongsong Zhu, and Limin Sun. TPLinker: Single-stage joint extraction of entities and relations through token pair linking. In Proceedings of the 28th International Conference on Computational Linguistics, pages 1572–1582, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics.

- [41] Zhiheng Yan, Chong Zhang, Jinlan Fu, Qi Zhang, and Zhongyu Wei. A partition filter network for joint entity and relation extraction. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 185–197, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics.

- [42] Xiangrong Zeng, Daojian Zeng, Shizhu He, Kang Liu, and Jun Zhao. Extracting relational facts by an end-to-end neural model with copy mechanism. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 506–514, 2018.

- [43] Ranran Haoran Zhang, Qianying Liu, Aysa Xuemo Fan, Heng Ji, Daojian Zeng, Fei Cheng, Daisuke Kawahara, and Sadao Kurohashi. Minimize exposure bias of seq2seq models in joint entity and relation extraction. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 236–246, 2020.

- [44] Dianbo Sui, Yubo Chen, Kang Liu, Jun Zhao, Xiangrong Zeng, and Shengping Liu. Joint entity and relation extraction with set prediction networks. arXiv preprint arXiv:2011.01675, 2020.

- [45] Minghao Hu, Yuxing Peng, Zhen Huang, Xipeng Qiu, Furu Wei, and Ming Zhou. Reinforced mnemonic reader for machine reading comprehension. arXiv preprint arXiv:1705.02798, 2017.

- [46] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, 2016.

- [47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

- [48] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. arXiv preprint arXiv:2107.13586, 2021.

- [49] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582–4597, Online, August 2021. Association for Computational Linguistics.

- [50] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of NAACL-HLT 2019: Demonstrations, 2019.

- [51] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR (Poster), 2015.

- [52] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, October 2020. Association for Computational Linguistics.

- [53] Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7871–7880, Online, July 2020. Association for Computational Linguistics.
