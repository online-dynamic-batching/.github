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

Online Dynamic Batching (ODB) forms token-budgeted batches at the
DataLoader/collate boundary, after the real training length is known. It is
designed for variable-length LLM and multimodal fine-tuning without changing
model code, optimizer logic, attention kernels, or dataset format.

| Entry | Use it for |
| --- | --- |
| [`online-dynamic-batching`](https://github.com/online-dynamic-batching/online-dynamic-batching) | Main pip package, API docs, integration guides, tests, and benchmarks |
| [`pip install online-dynamic-batching`](https://pypi.org/project/online-dynamic-batching/) | Install the released package |
| [Quickstart](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/quickstart.md) | First PyTorch DataLoader replacement |
| [Integration guides](https://github.com/online-dynamic-batching/online-dynamic-batching/tree/main/docs/integration-guides) | Pick PyTorch, Hugging Face Trainer, LLaMA-Factory, Accelerate, or Lightning |

## Project Map

These repositories are meant to be read as a small ecosystem: the main package
defines the API, and each example project shows one complete framework path.

| Repository | Role |
| --- | --- |
| [`online-dynamic-batching`](https://github.com/online-dynamic-batching/online-dynamic-batching) | Core package and documentation |
| [`odb-mm-mix-example`](https://github.com/online-dynamic-batching/odb-mm-mix-example) | Shared public MM-Mix-style data recipe and local TMDB utilities |
| [`odb-example-llamafactory`](https://github.com/online-dynamic-batching/odb-example-llamafactory) | LLaMA-Factory training and evaluation workflow |
| [`odb-example-hf-trainer`](https://github.com/online-dynamic-batching/odb-example-hf-trainer) | Hugging Face `Trainer` workflow |
| [`odb-example-accelerate`](https://github.com/online-dynamic-batching/odb-example-accelerate) | Accelerate custom-loop workflow |
| [`odb-example-lightning`](https://github.com/online-dynamic-batching/odb-example-lightning) | Lightning Trainer workflow |

## Choose One Framework Path

Pick one row. The examples are alternatives, not steps in a single checklist.

| Framework | ODB entry point | Example |
| --- | --- | --- |
| PyTorch loop | `odb.ODBDataLoader(...)` or `odb.apply(...)` | [`pytorch-loop.md`](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/pytorch-loop.md) |
| Hugging Face Trainer | `odb.integrations.hf` | [`odb-example-hf-trainer`](https://github.com/online-dynamic-batching/odb-example-hf-trainer) |
| LLaMA-Factory | `odb.integrations.llamafactory` | [`odb-example-llamafactory`](https://github.com/online-dynamic-batching/odb-example-llamafactory) |
| Accelerate | `odb.integrations.accelerate` | [`odb-example-accelerate`](https://github.com/online-dynamic-batching/odb-example-accelerate) |
| Lightning | `odb.integrations.lightning` | [`odb-example-lightning`](https://github.com/online-dynamic-batching/odb-example-lightning) |

## Paper

ODB is introduced in
[**Online Dynamic Batching with Formal Guarantees for LLM Training**](https://arxiv.org/abs/2606.19989)
by Dian Li, Zekun Wang, Yaoru Wang, and Jiahong Yan.

Links: [arXiv](https://arxiv.org/abs/2606.19989) ·
[PDF](https://arxiv.org/pdf/2606.19989) ·
[BibTeX](https://github.com/online-dynamic-batching/online-dynamic-batching#citation)

## Design Principles

- Observe real runtime lengths instead of relying on stale offline caches.
- Keep batching at the DataLoader boundary.
- Make distributed dynamic batching explicit through aligned grouping metadata.
- Keep examples framework-specific so users can copy the path closest to their stack.

Apache-2.0.
