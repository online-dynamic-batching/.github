# Online Dynamic Batching

Online Dynamic Batching builds DataLoader-side batching tools for variable-length
LLM and multimodal fine-tuning.

Our main package, **Online Dynamic Batching (ODB)**, observes training length
after the real input pipeline has run: preprocessing, chat templates,
tokenization, truncation, augmentation, and multimodal visual-token expansion.
It then forms token-budgeted batches online, so short examples get larger
batches and long examples get smaller batches without changing the model,
optimizer, attention kernel, or dataset format.

## Start Here

- [`online-dynamic-batching`](https://github.com/online-dynamic-batching/online-dynamic-batching):
  the PyTorch package with DataLoader integration, DDP alignment, trainer
  adapters, docs, tests, and public synthetic benchmarks.
- [Quickstart](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/quickstart.md):
  install the package and run a minimal PyTorch loop.
- [Integration guides](https://github.com/online-dynamic-batching/online-dynamic-batching/tree/main/docs/integration-guides):
  PyTorch loops, HuggingFace Trainer, LLaMA-Factory-style trainers, Accelerate,
  and Lightning.
- [Benchmark notes](https://github.com/online-dynamic-batching/online-dynamic-batching/blob/main/docs/benchmarks.md):
  reporting policy, public synthetic benchmark, and representative results.

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

## What We Care About

- Training-time length observability instead of stale offline length caches.
- Distributed training semantics that make variable local work explicit.
- Practical trainer adapters with a clear metadata and loss-scaling contract.
- Reproducible public validation before production-scale claims.
- Small, reviewable integrations that keep model code and kernels untouched.

## Ecosystem

For the public launch, the main repository carries the package, documentation,
examples, benchmark notes, and agent-assisted integration skill. Supporting
repositories should be split out only when they have independent value: a docs
site, a public benchmark harness, paper artifacts based only on public data, or
community-maintained recipes.

Apache-2.0.
