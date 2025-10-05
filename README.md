# Estimating LLM Benchmark Scores from Descriptions

🤗[Dataset](https://huggingface.co/datasets/jungsoopark/PRECOG) | 📄[Paper](https://arxiv.org/abs/2509.20645) 

## Introduction

**Official repository for the paper:** [Look Before You Leap: Estimating LLM Benchmark Scores from Descriptions](https://www.arxiv.org/abs/2509.20645)

_PRECOG provides a standardized testbed to study early difficulty estimation and planning (what to run, where to measure) without incurring full evaluation cost._


### Repo Overview
- **Data curation pipeline:** mines *description → performance* pairs from the literature (arXiv 2023–2024), with redaction and normalization.
- **Baseline predictors:** reasoning LLMs with a **leakage-resistant retrieval module** to forecast scores from text only.

### Benchmark ([PRECOG](https://huggingface.co/datasets/jungsoopark/PRECOG)🔮)
- **Task:** Given a redacted **natural-language description** of a task/evaluation setup, predict the **reported model score** *before* running any experiments.
- **Dataset:** A corpus of redacted descriptions paired with literature-reported scores, curated to study text-only “description→performance” forecasting.

## Notes

**Code will be available soon.**

## Citation

If you use our work or are inspired by our work, please cite our work:

```
@article{park2025look,
  title={Look Before you Leap: Estimating LLM Benchmark Scores from Descriptions},
  author={Park, Jungsoo and Mendes, Ethan and Stanovsky, Gabriel and Ritter, Alan},
  journal={arXiv preprint arXiv:2509.20645},
  year={2025}
}

@inproceedings{park2025can,
    title = "Can {LLM}s Help Uncover Insights about {LLM}s? A Large-Scale, Evolving Literature Analysis of Frontier {LLM}s",
    author = "Park, Jungsoo  and
      Kang, Junmo  and
      Stanovsky, Gabriel  and
      Ritter, Alan",
    year = "2025",
    booktitle = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.998/"
}
```
