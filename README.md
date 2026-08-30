# GLM-5.3-Flash on 8x A800 with vLLM

[English](README_EN.md) | 中文

这是一个**非官方社区部署方案**，用于在 `8 x NVIDIA A800-SXM4-80GB` 上通过 vLLM 部署 GLM-5.3-Flash。仓库包含固定源码基线、SM80 集成补丁、最终覆盖文件、Docker 构建脚本、1M 上下文启动脚本、多模态配置和压力测试。

> 本仓库不包含模型权重、API 密钥、OAuth 文件、OneAPI 数据库或预构建镜像。

## 已验证能力

- GLM-5.3-Flash FP8 MoE，`320B` 总参数、约 `18B` 激活参数
- `1,048,576` tokens 上下文
- OpenAI Chat Completions、Responses 和 Anthropic Messages 路由
- 图片输入：每个请求最多 1 张；视频默认关闭
- CUDA Graph：`FULL_AND_PIECEWISE`
- Prefix Cache 与 GLM KPool sparse attention
- `TP=4 + PP=2`，8 张 A800 全部参与
- 默认调度：`max-num-seqs=16`、`max-num-batched-tokens=16384`

## 测试环境

| 项目 | 实测值 |
|---|---|
| GPU | 8 x NVIDIA A800-SXM4-80GB, SM80 |
| GPU 拓扑 | 任意两卡之间 `NV8` |
| NVIDIA Driver | `575.57.08` |
| Docker Engine | `28.3.3` |
| 基础源码 | `wtdcode/vllm-backport@2674c8bb6d8799b32158c94bee33356d84772a2a` |
| GLM 集成来源 | `ZJY0516/vllm:glm-release@6f0369074d9f755917ee2d29c15809ea73bcbfba` |
| 基础镜像 | `lazymio/vllm-backport:latest-sm80` |
| 实测基础镜像 ID | `sha256:65a0299a7cfdbcfda382110157aa811055a54f9c0f3cd9a578ad008a0e52bdc3` |
| 最终镜像名 | `glm53-a800:sm80-v9-cudagraph` |
| 最终镜像大小 | 约 `17.76 GB`，不含权重 |
| 权重大小 | 约 `305.8 GiB` |

A100 同为 SM80，理论上可以运行，但本仓库只在 A800 上验证。H100/H800/其他架构需要重新编译并单独验证。

## 目录结构

```text
docker/          SM80 镜像构建文件
patches/         相对固定基线的完整集成补丁
overrides/       线上最终版本的 Python/Triton 覆盖文件
scripts/         准备源码、构建、启动、导入导出镜像
tests/           KPool、单流、并发、长上下文和图片测试
docs/            架构、基准和 OneAPI 配置说明
```

## 前置条件

建议准备：

- 8 张 80GB SM80 GPU，并安装 NVIDIA Container Toolkit
- 500GB 以上系统内存；当前测试机为约 512GB
- 至少 450GB 可用磁盘；构建期间建议更多
- Git、Docker、Python 3、`zstd`
- 已下载的 GLM-5.3-Flash 权重

模型权重遵循模型仓库自己的 MIT License，本仓库不会重新分发权重。

## 1. 获取模型

ModelScope：`ZhipuAI/GLM-5.3-Flash`

```bash
python3 -m pip install -U modelscope
modelscope download \
  --model ZhipuAI/GLM-5.3-Flash \
  --local_dir /opt/docker/models/GLM-5.3-Flash
```

确认以下文件存在：

```bash
test -s /opt/docker/models/GLM-5.3-Flash/model.safetensors.index.json
```

## 2. 构建镜像

```bash
git clone https://github.com/Mrzhiyao/glm53-a800-vllm.git
cd glm53-a800-vllm

./scripts/prepare-source.sh
./scripts/build-image.sh
```

可先校验仓库中的构建输入：

```bash
sha256sum -c MANIFEST.sha256
```

默认产物：

```text
glm53-a800:sm80-v9-cudagraph
```

构建脚本会：

1. 克隆固定的 `wtdcode/vllm-backport` 提交；
2. 应用 `patches/glm53-sm80.patch`；
3. 覆盖 `overrides/` 中已经在线验证的最终实现；
4. 只为 `SM80` 编译 vLLM CUDA 扩展；
5. 生成不含模型权重的运行镜像。

基础镜像标签是可变的。若 `latest-sm80` 与上表实测镜像 ID 不一致，请先按 [故障排查](#故障排查) 处理，不要默认认为新基础层仍然兼容。

## 3. 启动服务

默认配置对应本机最终实测档位：

```bash
MODEL=/opt/docker/models/GLM-5.3-Flash \
IMAGE=glm53-a800:sm80-v9-cudagraph \
./scripts/run-server.sh
```

主要参数：

```text
TP=4
PP=2
max_model_len=1048576
gpu_memory_utilization=0.80
max_num_seqs=16
max_num_batched_tokens=16384
image=1
video=0
```

需要更保守的档位时：

```bash
MAX_NUM_SEQS=12 \
MAX_NUM_BATCHED_TOKENS=8192 \
./scripts/run-server.sh
```

查看启动状态：

```bash
docker logs -f glm53-flash-a800
curl http://127.0.0.1:8010/v1/models
```

首次启动需要加载约 306GiB 权重并捕获 CUDA Graph，耗时数分钟是正常现象。

## 4. API 测试

文本：

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"glm-5.3-flash",
    "messages":[{"role":"user","content":"Reply with exactly OK."}],
    "max_tokens":64,
    "temperature":0
  }'
```

图片使用标准 OpenAI `image_url` 消息格式。GUI 客户端通常会在勾选 Vision/视觉能力后自动生成该格式，无需手写 Base64。

## 5. 压力测试

```bash
python3 tests/stress_single.py
python3 tests/stress_concurrent16.py
python3 tests/stress_long16.py
```

测试脚本默认访问 `http://127.0.0.1:8010`。

## 6. 从已部署机器导出镜像

当前已验证镜像名：

```text
glm53-a800:sm80-v9-cudagraph
```

在服务器执行：

```bash
sudo mkdir -p /opt/docker/exports

IMAGE=glm53-a800:sm80-v9-cudagraph \
OUT=/opt/docker/exports/glm53-a800-sm80-v9.tar.zst \
./scripts/export-image.sh
```

得到：

```text
/opt/docker/exports/glm53-a800-sm80-v9.tar.zst
/opt/docker/exports/glm53-a800-sm80-v9.tar.zst.sha256
```

从另一台电脑下载：

```bash
scp -P <ssh-port> <user>@<server>:/opt/docker/exports/glm53-a800-sm80-v9.tar.zst* .
```

导入目标服务器：

```bash
./scripts/import-image.sh ./glm53-a800-sm80-v9.tar.zst
docker image inspect glm53-a800:sm80-v9-cudagraph
```

镜像不包含模型权重。目标服务器仍需要单独准备约 306GiB 权重，并满足 SM80/CUDA 驱动条件。

## 实测结果

以下是本机测试，不是官方基准：

| 配置 | 结果 |
|---|---|
| eager 单流 | 约 `5.74 tok/s` |
| CUDA Graph 单流，512 tokens | `10.93s`，约 `46.8 tok/s` |
| 12 路热态，12x64 tokens | `4.90s`，聚合约 `157 tok/s` |
| 16 路热态，16x64 tokens | `3.37s`，聚合约 `304 tok/s` |
| 16 路长上下文 | 16x约50K prompt tokens，约 `61.9s` |
| 长上下文 KV 峰值 | 约 `30.7%` |
| 图片经 OneAPI | 正确识别，约 `3.52s` |

`16/16384` 启动画像：

- activation 峰值约 `2.75 GiB/卡`
- CUDA Graph 约 `0.91 GiB/卡`
- KV Cache 约 `261.8 万 tokens`
- 完整 1M 上下文容量约 `2.50x`

## 重要说明

- `max_num_seqs` 是调度器允许同时活动的 sequence 上限，不等于总 HTTP 连接数。
- `max_num_batched_tokens` 主要影响长提示词预填充，不会让每轮一次生成数千个 token。
- 16 个超长上下文可能触发 KV 抢占；请监控 `/metrics`。
- `GPU-Util=99%` 不代表 Tensor Core 已吃满。MoE、小 kernel、TP/PP 和 NCCL 通信都可能造成高 Util、低功率。
- 不要以接近 400W 为优化目标，应关注 tokens/s、TTFT、排队、错误和 P95 延迟。

## 故障排查

### `illegal memory access` 或 CUDA Graph 捕获失败

确认最终 `overrides/.../kpool_compress.py` 已覆盖到源码，并执行：

```bash
python3 tests/test_kpool_cudagraph_safe.py
```

### `Input token limit exceeded`

检查 `/v1/models` 返回的 `max_model_len` 是否仍为 `1048576`，并为输出和系统提示预留空间。

### 服务启动但请求很慢

检查：

```bash
curl -s http://127.0.0.1:8010/metrics | grep -E \
  'num_requests_(running|waiting)|kv_cache_usage|generation_tokens'
```

### 外层出现 502

本地 vLLM 和 OneAPI 可能仍在正常运行。若日志包含 `Broken pipe`，通常表示客户端或外层代理先断开，应检查最外层反向代理和客户端超时，而不是只修改 vLLM。

## OneAPI

OneAPI 频道可设置为：

```text
Base URL: http://127.0.0.1:8010
Models: GLM-5.3-Flash,glm-5.3-flash
```

详见 [docs/ONEAPI.md](docs/ONEAPI.md)。

## License 与来源

仓库中的 vLLM 派生代码和补丁遵循 Apache-2.0。GLM-5.3-Flash 权重遵循模型仓库的 MIT License。详见 [NOTICE.md](NOTICE.md)。
