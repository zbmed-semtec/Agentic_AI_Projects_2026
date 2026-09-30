# LinkBERT: Pretraining Language Models with Document Links

**Paper ID:** 08
**Authors:** Michihiro Yasunaga, Jure Leskovec, Percy Liang
**Year:** 2022
**Venue:** ACL 2022
**DOI:** 10.18653/v1/2022.acl-long.551
**Source:** https://aclanthology.org/2022.acl-long.551/

## Abstract

Language model pretraining captures various knowledge from text corpora, helping downstream tasks. However, existing methods such as BERT model a single document, and do not capture dependencies or knowledge that span across documents. In this work, we propose LinkBERT, an LM pretraining method that leverages links between documents, e.g., hyperlinks. Given a text corpus, we view it as a graph of documents and create LM inputs by placing linked documents in the same context. We then pretrain the LM with two joint self-supervised objectives: masked language modeling and document relation prediction. We show that LinkBERT outperforms BERT on various downstream tasks across two domains, including a biomedical domain pretrained on PubMed with citation links. The biomedical LinkBERT model sets new state-of-the-art results on various BioNLP tasks.
