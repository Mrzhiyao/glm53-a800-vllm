# Notices and upstream sources

This repository is an unofficial integration and is not endorsed by Z.AI, vLLM, NVIDIA, or the upstream maintainers listed below.

## vLLM-derived code

- Baseline: `wtdcode/vllm-backport`
- Commit: `2674c8bb6d8799b32158c94bee33356d84772a2a`
- License: Apache License 2.0

## GLM integration reference

- Repository/ref: `ZJY0516/vllm:glm-release`
- Commit: `6f0369074d9f755917ee2d29c15809ea73bcbfba`
- License: follows the corresponding vLLM-derived source files

## Base container

- Image: `lazymio/vllm-backport:latest-sm80`
- Tested image ID: `sha256:65a0299a7cfdbcfda382110157aa811055a54f9c0f3cd9a578ad008a0e52bdc3`
- The base tag is mutable; users must verify compatibility before rebuilding.

## Model

- Model: GLM-5.3-Flash
- Publisher: Z.AI / ZhipuAI
- Model license: MIT
- Model weights are not included or redistributed by this repository.

The files under `patches/` and `overrides/` are derived from the upstream projects above and remain subject to their original license terms.
