# Deployment architecture

## Parallel layout

The tested service uses TP4+PP2:

```text
PP0: TP0 TP1 TP2 TP3
PP1: TP0 TP1 TP2 TP3
```

The layer partition is set to `24,21`. Every generated token traverses both pipeline stages. TP8+PP1 was tested earlier and was slower for this workload, so it is not the default.

## Why CUDA Graph matters

The eager configuration generated approximately 5.74 tokens/s in the original single-stream test. After restoring CUDA Graph capture with capture-safe KPool kernels, the same model generated approximately 47 tokens/s.

The bounded KPool implementation keeps invalid physical tail slots from forming out-of-range addresses while avoiding data-dependent `torch.nonzero` output shapes during graph capture.

## Scheduler settings

- `max_num_seqs` limits active scheduler sequences, not total HTTP connections.
- `max_num_batched_tokens` limits tokens scheduled per engine iteration, primarily affecting prompt prefill.
- Extra requests remain in the capacity queue.
- KV availability can reduce the number of simultaneously active long-context requests below `max_num_seqs`.

## Multimodal

The official checkpoint contains approximately 1.05GiB of visual-tower weights. The deployment enables one image per prompt and disables video. The text model continues to use the SM80 Triton sparse MLA backend.
