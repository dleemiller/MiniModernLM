# MiniModernLM

This repository builds on the [MiniLMv2](https://aclanthology.org/2021.findings-acl.188.pdf) distillation technique, for distilling ModernBERT.
Code is adapted from a fork of [minilmv2.bb](https://github.com/bloomberg/minilmv2.bb).

We add PCA token embedding initialization from the teacher model, and convert training for ModernBERT architecture.

This code uses the [Hugging Face Transformers](https://github.com/huggingface/transformers) library.

## Menu

- [Quick Start](#quick-start)
- [Hyperparameters](#hyperparameters)
- [Results](#results)
- [License](#license)

## Quick Start

Start the distillation by running `run.sh`.

## Hyperparameters

For distillation, the `run.sh` script contains the hyperparameters used to distill models and can be modified for other settings as required.
The `run_eval.sh` script contains the evaluation hyperparameters and we hypertune our models over different learning rates (`1e-5` - `4e-5`) and num epochs (`3`,`5`,`10`).

## Citation
The MiniLMV2 technique was originally proposed by:
```
  @inproceedings{wang-etal-2021-minilmv2,
    title = "{M}ini{LM}v2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers",
    author = "Wang, Wenhui  and
      Bao, Hangbo  and
      Huang, Shaohan  and
      Dong, Li  and
      Wei, Furu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.188",
    doi = "10.18653/v1/2021.findings-acl.188",
    pages = "2140--2151",
}
```

## Results

Coming soon...


## License

See LICENSE
