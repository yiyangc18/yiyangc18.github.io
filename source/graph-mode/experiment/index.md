---
title: "Qwen3-8B 图模式实测"
layout: page
---

[返回图模式文章](/2026/09/30/graph-mode/)

# Qwen3-8B 图模式实测

实验日期：2026-09-29。通过 npu-remote 在远端 Ascend A3 环境完成，以下为每种模式 3 个独立进程运行的中位数；方括号为最小值、最大值。

## 结果

| 模式 | Decode 平均时延 ms/步 | Forward 吞吐 token/s | 实际生成吞吐 token/s | Forward 相对 Eager |
|---|---:|---:|---:|---:|
| Eager（不开图） | 63.18 [62.86, 63.18] | 15.53 [15.53, 15.60] | 13.61 [13.59, 13.64] | 1.00× |
| TorchAir GE Graph | 17.40 [17.13, 17.85] | 53.81 [52.57, 54.91] | 40.21 [39.62, 40.96] | 3.46× |
| NPU Graph EX | 23.86 [23.84, 23.86] | 39.77 [39.70, 39.78] | 29.38 [29.29, 29.82] | 2.56× |

本次短输入、batch=1 工作负载下，GE Graph 的 forward 吞吐为 Eager 的 3.46 倍，NPU Graph EX 为 2.56 倍；两者实际生成吞吐分别提升到 2.96 倍和 2.16 倍。该结论只覆盖本次固定 prompt 和配置。

![三种模式性能对比](/img/graph-mode/qwen3_8b_comparison.png)

| 模式 | Prefill ms | Decode 总时间 s | 预热 wall s | 加载及预热 wall s | 整进程 wall s |
|---|---:|---:|---:|---:|---:|
| Eager（不开图） | 77.66 [77.00, 80.16] | 4.04 [4.02, 4.04] | 5.14 [5.09, 6.32] | 29.08 [28.87, 31.91] | 47.07 [46.49, 49.46] |
| TorchAir GE Graph | 74.91 [69.46, 75.97] | 1.11 [1.10, 1.14] | 47.74 [47.23, 63.04] | 71.58 [70.95, 86.81] | 86.02 [85.42, 101.04] |
| NPU Graph EX | 82.03 [82.01, 86.00] | 1.53 [1.53, 1.53] | 41.93 [41.77, 42.63] | 62.52 [62.41, 63.64] | 76.69 [76.52, 77.09] |

预热包含教程的一步 prefill 和一步 decode，图后端在其中触发编译、捕获等工作，因此它不是纯编译时间。整进程 wall 包括 Python 启动、加载、预热/编译、正式生成及退出。每轮是新进程，`enable_cache_compile=False`，但未清理 CANN 算子缓存或 OS 文件缓存，不能称为完全冷缓存启动；NPU Graph EX 的成功首轮发生在一次失败编译尝试之后。

## 测量口径

- 模型：ModelScope `Qwen/Qwen3-8B` 原始 BF16 权重，位于远端 `/mnt/workspace/vllm_workspace/models/Qwen3-8B`。
- 硬件：`npu-smi` 报告 Ascend910，板卡 `IT22HMDA_4_S`；运行环境 SoC 标识 `ascend910_9391`，recipe 使用 A3 配置。只使用逻辑 NPU 0（板卡 5、芯片 0 / 物理 ID 10），64 GiB HBM。
- 软件：PyTorch `2.10.0+cpu`、torch_npu `2.10.0.post4`（自带 TorchAir/npugraph_ex）、CANN `9.1.0`、Transformers `5.14.1`、Python 3.12。完整版本见 `results/environment.json` 和 `environment.freeze.txt`。这与 notebook 标注的 PyTorch 2.8 / CANN 9.0 环境不同。
- TP=1、batch=1，输入上限 256 token，默认 attention prompt 实际 50 token，关闭 thinking，返回 64 token。
- 使用 notebook 自带 recipe 和相同 YAML，三组仅 `exe_mode` 不同。关闭 profiler、编译缓存、static kernel、显式 RMSNorm/AddRMSNorm 融合及权重预取；其他 recipe 默认项相同。
- 图模式只编译 decode，prefill 仍为 Eager。实际 TASK_QUEUE_ENABLE：Eager/GE=2，NPU Graph EX=1。
- 每次初始化沿用 recipe 的预热；正式生成执行 1 步 prefill + 64 步 decode，返回前 64 个输出 token。这是教程 offline scheduler 的既有行为，本实验没有修改。
- Forward 吞吐 = 实际返回的 64 token ÷ 同步测得的 prefill+decode forward 总时间；decode 平均时延 = 全部 64 个 decode 步的总时间 ÷ 64。保留全部步，不剔除离群点。原日志的 `decode average inference time` 会再跳过一步并过滤离群点，所以与本表略有不同。
- 实际生成吞吐 = 64 ÷ `llm.generate()` wall time，含调度、采样、逐步日志及最终文本解码，但不含模型加载、预热和首次编译。不是服务端并发吞吐或网络 TTFT。
- 顺序：Eager→GE→NPU Graph EX；GE→NPU Graph EX→Eager；NPU Graph EX→Eager→GE。各轮使用新进程，三次测量范围不等同于置信区间。
- 测试期间已暂停既有 vLLM 服务；既有 NPU keepalive 进程保留，CPU 为共享宿主机。

## 输出与后端验证

每次成功运行必须满足：进程退出码 0、64 个有效输出 token、65 个正式 forward 计时、正常 length 结束；图模式还必须出现编译触发和预热成功日志。

| 运行 | 与 Eager 首轮文本相似度 | token ID 完全相同 |
|---|---:|---|
| eager_r1 | 1.0000 | 是 |
| eager_r2 | 1.0000 | 是 |
| eager_r3 | 1.0000 | 是 |
| ge_graph_r1 | 0.9852 | 否 |
| ge_graph_r2 | 0.9852 | 否 |
| ge_graph_r3 | 0.9852 | 否 |
| npugraph_ex_r1 | 0.9852 | 否 |
| npugraph_ex_r2 | 0.9852 | 否 |
| npugraph_ex_r3 | 0.9852 | 否 |

两种图后端在三轮内生成的 token ID 相同；相对 Eager 均有 2/64 个 token 不同，文本相似度 98.52%，超过 notebook 使用的 0.95 阈值。

文本相似度使用 `SequenceMatcher`，仅为本条 prompt 的粗粒度生成一致性检查，不代表困惑度、精度评测或逐 token 数值等价。原始文本和 token ID 保存在各轮 `raw.json`。

## 启动问题与处理

NPU Graph EX 最初因 TASK_QUEUE_ENABLE=2 无法捕获图。原因是 recipe 的 `ModelConfig._validate()` 在外部已经设为 1 时会进入 else 分支改回 2。随后严格沿用 notebook 的 infer.sh 行为：启动时先设为 2，交由 recipe 自动改为 1，三轮成功。未修改 recipe 源码；失败尝试独立保存在 `failed_attempts/`，不纳入统计。

## 复现与制品

参考 notebook：`/mnt/workspace/cann-learning-hub/tutorials/llm_inference/qwen3_8b/07_npu_graph_optimization.ipynb`。来源仓库 HEAD：`738809a238ffd72c3b170f9c7a801293e6dcde1b`。实际使用源文件的 SHA256 见 `source_manifest.json`，原始源码快照见 `tutorial-src.tar.gz`。

远端实验目录：`/home/developer/graphmode-experiment`。脚本通过相同的 Python inference 入口运行；`entrypoint.py` 仅包装初始化、预热、生成函数以记录 wall time 和原始返回值，未替换模型/编译实现。

```bash
# 在远端实验目录，选择新的 results 目录，或先归档已有 results。
# run.py 会跳过已有 status=0 且 raw.json 存在的运行。
cd /home/developer/graphmode-experiment
bash start.sh
# 本地汇总、重画图：
python analyze.py
python plot.py
python report.py
```

- `results/runs.csv`：全部 9 轮数据。
- `results/summary.json`：中位数、范围、文本一致性。
- `results/<mode>_r<n>/`：YAML、完整 stdout、实际 token ID/逐步时延、进程状态。
- `comparison.png/.svg/.pdf`：可用于演示稿的独立图表。
- `service/`：vLLM 暂停前和恢复后的检查记录。
- 模型在用户指定的 `/mnt/workspace`，随 NPU 实例释放可能清空；实验代码及日志另存于远端 home，并回传本地仓库。

## vLLM 恢复验证

已于 2026-09-29T20:56:40+0800 按原参数恢复 `qwen3.5-9b` 服务（TP=2，max_model_len=32768，端口 8000）。`/health` 返回 200，实际 chat completion 返回 `OK`。检查记录见 `service/verification.json` 和 `service/smoke_response.json`。
