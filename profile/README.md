<p align="center">
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching">
    <img alt="Main repository" src="https://img.shields.io/badge/main-online--dynamic--batching-2f6fef">
  </a>
  <a href="https://pypi.org/project/online-dynamic-batching/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/online-dynamic-batching.svg">
  </a>
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-1f883d">
  </a>
  <a href="https://arxiv.org/abs/2606.19989">
    <img alt="arXiv paper" src="https://img.shields.io/badge/arXiv-2606.19989-b31b1b">
  </a>
</p>

<h1 align="center">Online Dynamic Batching</h1>

<h2 align="center">Online dynamic batching for faster LLM/VLM fine-tuning</h2>

<p align="center">
  <img alt="Up to 4.4x faster" src="../assets/speedup-badge-wide.svg" width="560">
</p>

## Start Here

New to ODB? Start with the main package, then open the example closest to your
training stack.

| Entry | Use it for |
| --- | --- |
| [`online-dynamic-batching`](https://github.com/online-dynamic-batching/online-dynamic-batching) | Install ODB, read the API, and choose an integration path |
| [`pip install online-dynamic-batching`](https://pypi.org/project/online-dynamic-batching/) | Install the released package |
| [Quickstart](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/quickstart.md) | Try the minimal PyTorch loop and understand the ODB-ready batch contract |
| Example projects | Run a framework-specific MM-Mix workflow |

## Project Map

The organization is split into a core package and small framework-specific
examples.

| Repository | Role |
| --- | --- |
| [`online-dynamic-batching`](https://github.com/online-dynamic-batching/online-dynamic-batching) | Core package, integration guides, API docs, tests, and benchmark notes |
| [`odb-mm-mix-example`](https://github.com/online-dynamic-batching/odb-mm-mix-example) | Shared public MM-Mix-style data recipe and local TMDB utilities |
| [`odb-example-llamafactory`](https://github.com/online-dynamic-batching/odb-example-llamafactory) | LLaMA-Factory training and evaluation workflow |
| [`odb-example-hf-trainer`](https://github.com/online-dynamic-batching/odb-example-hf-trainer) | Hugging Face `Trainer` workflow |
| [`odb-example-accelerate`](https://github.com/online-dynamic-batching/odb-example-accelerate) | Accelerate custom-loop workflow |
| [`odb-example-lightning`](https://github.com/online-dynamic-batching/odb-example-lightning) | Lightning Trainer workflow |

## Paper

ODB is introduced in
[**Online Dynamic Batching with Formal Guarantees for LLM Training**](https://arxiv.org/abs/2606.19989)
by Dian Li, Zekun Wang, Yaoru Wang, and Jiahong Yan.

Links: [arXiv](https://arxiv.org/abs/2606.19989) ·
[PDF](https://arxiv.org/pdf/2606.19989) ·
[BibTeX](https://github.com/online-dynamic-batching/online-dynamic-batching#citation)

## Design Principles

- Observe real runtime lengths instead of relying on stale offline caches.
- Keep batching at the DataLoader/collate boundary.
- Make distributed dynamic batching explicit with aligned grouping metadata.
- Keep examples framework-specific so users can copy the path closest to their stack.

Apache-2.0.
