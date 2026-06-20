<h1 align="center">Online Dynamic Batching</h1>

<h2 align="center">Faster LLM/VLM training with one DataLoader line</h2>

<p align="center">
  <img alt="Up to 2.5x faster" src="https://img.shields.io/badge/up%20to-2.5x%20faster-d73a49?style=for-the-badge">
</p>

<p align="center">
  <code>DataLoader(...)</code> → <code>odb.ODBDataLoader(...)</code>
</p>

<p align="center">
  Online dynamic batching for LLM and multimodal fine-tuning: faster batches,
  real post-preprocessing lengths, and no model or optimizer changes.
</p>

<p align="center">
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching">
    <img alt="Main repository" src="https://img.shields.io/badge/repo-online--dynamic--batching-2f6fef">
  </a>
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-1f883d">
  </a>
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/pyproject.toml">
    <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-3776ab">
  </a>
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching/tree/main/docs/integration-guides">
    <img alt="Integrations" src="https://img.shields.io/badge/integrations-PyTorch%20%7C%20HF%20%7C%20Accelerate%20%7C%20Lightning-6f42c1">
  </a>
</p>

<p align="center">
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/pytorch-loop.md">
    <img alt="PyTorch DataLoader" src="https://img.shields.io/badge/PyTorch-DataLoader-EE4C2C?logo=pytorch&logoColor=white">
  </a>
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/hf-trainer.md">
    <img alt="HuggingFace Trainer" src="https://img.shields.io/badge/HuggingFace-Trainer-FFD21E?logo=huggingface&logoColor=black">
  </a>
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/llamafactory.md">
    <img alt="LLaMA-Factory adapter" src="https://img.shields.io/badge/LLaMA--Factory-adapter-4B8BBE?logo=github&logoColor=white">
  </a>
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/accelerate.md">
    <img alt="Accelerate loops" src="https://img.shields.io/badge/Accelerate-loops-7C3AED?logo=huggingface&logoColor=white">
  </a>
  <a href="https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/lightning.md">
    <img alt="Lightning Trainer" src="https://img.shields.io/badge/Lightning-Trainer-792EE5?logo=lightning&logoColor=white">
  </a>
</p>

<p align="center">
  <img
    src="../assets/odb-banner.svg"
    alt="Online Dynamic Batching project banner"
    width="760"
  >
</p>

## The Project

**Online Dynamic Batching (ODB)** forms token-budgeted batches online, at the
DataLoader/collate boundary. Short examples get larger batches, long examples
get smaller batches, and the model, optimizer, attention kernel, and dataset
format stay in place.

```python
dataloader = odb.ODBDataLoader(..., token_budget=16384)  # replaces DataLoader(...)
```

Most training stacks decide batch shape before the final input length is known.
ODB moves that decision to the point where the length is already observable.

| Start here | What you get |
| --- | --- |
| [`online-dynamic-batching`](https://github.com/online-dynamic-batching/online-dynamic-batching) | PyTorch package, trainer adapters, docs, tests, examples, and synthetic benchmarks |
| [Quickstart](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/quickstart.md) | One-line DataLoader replacement and first PyTorch loop |
| [Choose one integration path](https://github.com/online-dynamic-batching/online-dynamic-batching/tree/main/docs/integration-guides) | Pick the adapter that matches your stack: PyTorch loops, HuggingFace Trainer, LLaMA-Factory/LLaVA-Factory, Accelerate, or Lightning |
| [Benchmark notes](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/benchmarks.md) | Reporting policy, public synthetic benchmark, and representative results |

## Choose One Integration Path

Pick the row that matches your training stack. These are alternatives, not a
checklist.

| Framework | ODB entry point |
| --- | --- |
| [PyTorch loops](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/pytorch-loop.md) | `ODBDataLoader(...)` or `odb.apply(dataloader, ...)` |
| [HuggingFace Trainer](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/hf-trainer.md) | `odb.integrations.hf.configure_trainer(...)` and `ODBTrainerMixin` |
| [LLaMA-Factory / LLaVA-Factory](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/llamafactory.md) | `odb.integrations.llamafactory.configure_trainer(...)` |
| [Accelerate](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/accelerate.md) | `odb.integrations.accelerate.configure_accelerator(...)` |
| [Lightning](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/integration-guides/lightning.md) | `odb.integrations.lightning.ODBLightningCallback` |

## Install

```bash
pip install online-dynamic-batching
```

For HuggingFace Trainer and LLaMA-Factory-style adapters:

```bash
pip install "online-dynamic-batching[hf]"
```

## Minimal Use

```python
import odb

dataloader = odb.ODBDataLoader(
    dataset,
    token_budget=16384,
    batch_size=1,
    num_workers=4,
    prefetch_factor=64,
    collate_fn=collate_fn,
    loss_scaling="exact",
)

for batch in dataloader:
    info = odb.pop_step_info(batch, loss_scaling="exact")
    loss = model(**batch).loss * info.loss_scale
    loss.backward()
```

## Design Values

- Observe real lengths at training time instead of trusting stale length caches.
- Keep batching at the DataLoader boundary, away from model and kernel code.
- Make distributed variable work explicit with aligned grouping and step info.
- Treat loss scaling, emitted-sample accounting, and trainer stopping semantics
  as first-class integration contracts.
- Keep public validation reproducible before making production-scale claims.

## Ecosystem Shape

The main repository carries the package, documentation, examples, benchmark
notes, and agent-assisted integration skill. Supporting repositories will be
split out only when they have independent value: a docs site, a public benchmark
harness, public-data paper artifacts, or community-maintained recipes.

Apache-2.0.
