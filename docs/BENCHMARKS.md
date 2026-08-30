# Benchmarks

These are engineering smoke benchmarks from one 8xA800 host, not official model benchmarks.

| Test | Result |
|---|---|
| Eager single stream | ~5.74 tok/s |
| CUDA Graph, one 512-token completion | 10.93s, ~46.8 tok/s |
| 12 concurrent, warm, 64 tokens each | 4.90s, ~157 aggregate tok/s |
| 16 concurrent, warm, 64 tokens each | 3.37s, ~304 aggregate tok/s |
| 16 concurrent x ~50K prompt tokens | 61.9s total |
| Long-context peak KV usage | ~30.7% |
| OneAPI image request | Correct answer in ~3.52s |

## 16/16384 startup profile

- Peak activation: approximately 2.75GiB per PP0 GPU
- CUDA Graph: approximately 0.91GiB per PP0 GPU
- KV cache: approximately 2.618 million tokens
- Full 1M-context concurrency estimate: 2.50x

## Interpretation

High aggregate throughput does not imply that each request is faster. At high concurrency, compute is shared among more active sequences. Always measure queue time, TTFT, inter-token latency, aggregate throughput, KV usage, preemption, and errors together.
