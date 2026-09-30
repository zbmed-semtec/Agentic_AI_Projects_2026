# Self-Alignment Pretraining for Biomedical Entity Representations

**Paper ID:** 07
**Authors:** Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, Nigel Collier
**Year:** 2021
**Venue:** NAACL-HLT 2021
**DOI:** 10.18653/v1/2021.naacl-main.334
**Source:** https://aclanthology.org/2021.naacl-main.334/

## Abstract

Despite the widespread success of self-supervised learning via masked language models (MLM), accurately capturing fine-grained semantic relationships in the biomedical domain remains a challenge. This is of paramount importance for entity-level tasks such as entity linking where the ability to model entity relations (especially synonymy) is pivotal. To address this challenge, we propose SapBERT, a pretraining scheme that self-aligns the representation space of biomedical entities. We design a scalable metric learning framework that can leverage UMLS, a massive collection of biomedical ontologies with 4M+ concepts. In contrast with previous pipeline-based hybrid systems, SapBERT offers an elegant one-model-for-all solution to medical entity linking, achieving a new state-of-the-art on six medical entity linking benchmarking datasets. In the scientific domain, we achieve strong results even without task-specific supervision.
