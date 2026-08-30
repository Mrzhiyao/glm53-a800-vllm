# Changelog

## 0.1.0 - 2026-08-30

- Pin the SM80 vLLM backport and GLM integration commits.
- Add A800-only CUDA extension build.
- Add Triton sparse MLA overrides for A800.
- Add bounded KPool tail-cache kernels compatible with CUDA Graph capture.
- Restore CUDA Graph acceleration while retaining KPool bounds checks.
- Enable 1,048,576-token context and one-image multimodal requests.
- Add TP4+PP2 launch profiles, with 16/16384 as the tested high-throughput default.
- Add single-stream, 16-way, long-context, KPool, and image tests.
- Add Docker image export/import instructions.
