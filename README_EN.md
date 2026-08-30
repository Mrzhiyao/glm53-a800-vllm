# GLM-5.3-Flash on 8x A800 with vLLM

This is an **unofficial community deployment recipe** for serving GLM-5.3-Flash on eight NVIDIA A800-SXM4-80GB GPUs.

It includes a pinned vLLM backport baseline, the GLM/SM80 integration patch, final runtime overrides, Docker build scripts, a 1M-context launch profile, image input, CUDA Graph fixes, and stress tests. Model weights and credentials are not included.

## Tested profile

- 8x A800 80GB, all GPU pairs connected by NV8
- Driver 575.57.08
- TP=4, PP=2
- Context length: 1,048,576
- `max_num_seqs=16`
- `max_num_batched_tokens=16384`
- One image per request, video disabled
- Final image: `glm53-a800:sm80-v9-cudagraph`, approximately 17.76GB without weights

## Quick start

```bash
git clone https://github.com/Mrzhiyao/glm53-a800-vllm.git
cd glm53-a800-vllm
./scripts/prepare-source.sh
./scripts/build-image.sh

MODEL=/opt/docker/models/GLM-5.3-Flash \
IMAGE=glm53-a800:sm80-v9-cudagraph \
./scripts/run-server.sh
```

See the Chinese [README](README.md) for complete hardware requirements, model download, benchmarks, image export/import, OneAPI, tuning, and troubleshooting.

## Compatibility

A100 is also SM80 and is expected to work, but only A800 has been tested. H100/H800 and other compute capabilities require a separate rebuild and validation.

## License

Derived vLLM code is Apache-2.0. GLM-5.3-Flash weights are MIT licensed by the model publisher and are not redistributed here.
