# Full-Vocab OPD：vLLM Teacher 迁移交付报告

工作目录：`/root/autodl-tmp/verl-full-vocab-opd`。veRL baseline 为
`3fdf2b548882a8aeccf0842c2150d9c9f1790a55`，保留既有 Student 实现后完成此次迁移。
实现与验证阶段未 commit/push；随后按用户要求发布至 `Muradil-mamat-211/HyOPD`
的 `full-vocab-vllm-teacher` 分支。保留 `verl-latest` 和原 `HyOPD` 工作区的既有改动。
迁移前文件备份：`/root/autodl-tmp/verl-full-vocab-opd-before-vllm.tar.gz`。

## 1–2. 文件清单及具体修改

下表描述相对上一版 Teacher-FSDP2 实现的本次修改。仓库 git diff 同时含上一版
未提交的 Student 实现；不能将所有 git diff 都理解为本次重写。

| 文件 | 本次变更 |
|---|---|
| `verl/experimental/teacher_loop/teacher_manager.py` | Full-Vocab 请求参数；原生 client 异步调用；检查 request identity、CPU/detach、L×V；CPU 温度变换及 cache dtype；Full 分支返回 `(None, teacher_logprobs)` |
| `verl/experimental/teacher_loop/teacher_model.py` | 为各 Teacher client 复制其实际 inference backend，支持 Student SGLang + Teacher vLLM；沿用原生初始化 |
| `verl/workers/rollout/full_vocab_logprobs.py`（新增） | 解码 vLLM full/flat prompt scores，按词表 ID 排列、完整性检查、单次 causal shift、request identity |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `prompt_logprobs=-1` 时调用 Full decoder，从实际 Teacher model config 获取 V；原有 Top-K extractor 保留 |
| `verl/experimental/agent_loop/agent_loop.py` | `_compute_teacher_logprobs` 支持无 teacher_ids 的 full 输出 |
| `verl/trainer/ppo/v1/agent_loop_tq.py` | 已完成 trajectory 的 Teacher scoring 与 reward 并发；多输出逐条 scoring；CPU 右补零到统一 T×V 后写原生 TQ；不新增 queue |
| `verl/utils/transferqueue_utils.py` | 原生 async TQ 读取边界，将 Full-Vocab 等长 nested 字段恢复为 dense CPU；Top-K 不变 |
| `verl/trainer/distillation/full_vocab_cache.py` | 删除旧 budget/jagged cache helpers；改为 CPU dense 恢复、右 padding、microbatch 输入搬运、loss 后置 Teacher H2D 及 action-row 对齐 |
| `verl/trainer/distillation/full_vocab.py` | 改为 Full vLLM mode 判断及 Student/tokenizer/RAM 验证；删除 Teacher TrainingWorker 配置构建和禁用 AgentLoop hook 的适配器 |
| `verl/trainer/distillation/losses.py` | 独立 Full registry 读取既有 Student full output 和当前 dense Teacher microbatch；仍调用原算法；明确禁止 full 进入 compute_topk_loss |
| `verl/workers/engine/fsdp/transformer_impl.py` | 删除 Teacher 专属 CPU offload 分支；Student 当前 forward 前保留 Teacher CPU 字段，forward 后由 loss 搬当前 microbatch；保留 Student full output/autograd 处理 |
| `verl/workers/engine_workers.py` | 删除 Teacher 专属冻结/optimizer 检查、inference 包装和 runtime vocab metadata 方法；保留原 Student TokenMetric 聚合 |
| `verl/trainer/ppo/v1/trainer_base.py` | 删除 Teacher TrainingWorker 注册/初始化、driver batch scoring、预算/释放调度及 Teacher DP padding；恢复 MultiTeacherModelManager；Student update flags 保留 |
| `verl/trainer/main_ppo.py` | 恢复原生 Teacher client 注入 AgentLoop；Full mode 校验 V1 |
| `verl/trainer/ppo/padding_utils.py` | Full synthetic padding 保持 Teacher dense 容量且监督 mask 为零；Top-K 保留原行为 |
| `verl/workers/config/distillation.py` | `full_vocab_vllm` 复用 execution_mode；删除 Teacher engine/model 配置，teacher_cache 仅 dtype；验证原生 vLLM backend/max_logprobs/raw scores 和资源分配 |
| `verl/trainer/config/distillation/distillation.yaml` | 移除 FSDP Teacher 配置；保留算法默认值和 CPU dtype、原生 teacher_models.inference |
| `examples/on_policy_distillation_trainer/full_vocab_adaptive_opd.yaml` | 改为原生 vLLM Teacher TP/DP/chunked-prefill/max_logprobs；移除 Teacher FSDP microbatch/budget 环境变量 |
| `tests/full_vocab_opd/test_vllm_teacher.py`（新增） | full/flat 协议、vocab 完整性、原生 Manager/client 并发乱序请求、identity/温度/CPU/no IDs |
| `tests/full_vocab_opd/test_dense_teacher_data.py`（新增） | causal/action rows、dynamic microbatch reorder、后置 micro H2D、TQ nested 恢复、synthetic padding |
| `tests/full_vocab_opd/test_agent_teacher_flow.py`（新增） | 实际 AgentLoop hook + Manager/client，确认 reward 尚未结束就发起各 trajectory 请求；CPU dense 写入；混合 rollout backend |
| `tests/full_vocab_opd/test_engine_chain.py` | CPU 小 Llama reference Teacher 接现有 Student 算子/engine/loss/backward/grad clip/SGD update；移除 FSDP Teacher stub |
| `tests/full_vocab_opd/test_yaml_config.py` | 验证原生 Teacher schema、环境变量和 tokenizer 契约 |
| `tests/full_vocab_opd/test_transfer_queue_integration.py` | real Ray/TQ dense put/get/native bridge/reorder/slice/clear opt-in 用例 |
| `tests/full_vocab_opd/test_v1_gpu_smoke.py` | real native vLLM Teacher + Student FSDP2 V1 单步；Student 参数/grad/loss 检查；无 Teacher TrainingWorker |
| `patches/liger/README.md` | 安装说明明确 patch 仅供 Student 使用；算子 patch 内容未改 |
| `docs/full_vocab_opd.md`、本报告、`full_vocab_opd_implementation_plan.md`、`full_vocab_opd_validation.txt` | 当前数据流、迁移审计、运行命令及实测边界 |

`coverage_adaptive_loss.py`、`torch_functional.py`、四文件 Liger patch，以及
`dense_common.py/qwen2_vl.py/qwen3_vl.py/qwen3_5.py/glm4v.py` 的 SHA256 均与本次
修改前快照一致。Student 算子与数学目标未重写。

## 3. 删除的上一版 Teacher-FSDP2 路线

已删除 `build_teacher_worker_config`、`agent_loop_config`、Teacher 专属
TrainingWorker/reset/infer 包装、`get_full_vocab_metadata`、driver
`_compute_teacher_full_log_probs`、`teacher_response_to_cpu`、`TeacherCacheBudget`、
`teacher_full_log_probs`、缓存 reservation/release/checkpoint guard、Teacher DP LCM
padding 及其配置。旧 `test_teacher_config.py`、`test_cache_alignment.py`、
`test_v1_flow.py` 已由当前路径测试替代，旧版本保存在上述备份。

`teacher_engine`、`teacher_model_config`、`full_vocab_fsdp2`、Teacher fsdp_size、
infer_micro_batch_size、infer_max_token_len、RAM max_cache_bytes/pin_memory 等旧配置
不再作为当前 Full mode 的参数。`teacher_cache` 名称保留，只含 CPU dtype。
原来的 `Role.TeacherModel/teacher_pool` 保留用于原生 vLLM replicas；不会初始化第二套
Teacher FSDP Worker。项目未新增 TeacherScoringQueue/TeacherScoringBatcher。

## 4. 实际 Teacher request 参数

```python
sampling_params = {
    "max_tokens": 1,
    "temperature": 1.0,
    "prompt_logprobs": -1,
    "flat_logprobs": True,
    "detokenize": False,
}
# Teacher engine:
engine_kwargs = {"vllm": {"max_logprobs": -1}}
logprobs_mode = "raw_logprobs"
```

整条 trajectory 是一次请求。一个生成 token 是原生 scoring API 占位，输出被忽略。
vLLM 自己拼接多个 prefill chunk 的完整 collection，应用仅接最终 request result。
API 审计依据是项目 pin 的 vLLM 0.32.0
[SamplingParams](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/sampling_params.py)、
[FlatLogprobs](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/logprobs.py)、
[prompt processing](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/engine/logprobs.py)。

## 5. vLLM → CPU `[B,T,V]` 的真实路径

```text
AgentLoopWorkerTQ._agent_loop_postprocess（每个已完成 trajectory）
 -> AgentLoopWorker._compute_teacher_logprobs
 -> AsyncTeacherLLMServerManager.compute_teacher_logprobs_single
 -> LLMServerClient.generate -> native vLLM server.generate
 -> full_vocab_prompt_fields -> extract_full_vocab_prompt_logprobs
 -> CPU FP32 [L,V]，按 vocab ID 排序并 shift 一次
 -> native TokenOutput.extra_fields，按 request/session identity 返回
 -> Manager 检查 CPU/detach/L/V，CPU log_softmax(raw/T)，cast cache dtype
 -> AgentLoop extra_fields.teacher_logprobs，无 teacher_ids
 -> pad_teacher_logprobs，CPU [T,V]
 -> 原生 TQ 写同一 trajectory key
 -> 原生 tqbridge 读取，restore_dense_teacher_logprobs，CPU [B,T,V]
```

TQ 0.1.11 实测会将等长字段读成 nested；增加读取边界的 dense 恢复正是针对这个
实际行为。生命周期继续使用原生 TQ key 管理，未引入额外缓存系统。

## 6. 当前 Student microbatch 如何取 Teacher

TensorDict 中的 Teacher 和 sample fields 同时通过原生 reorder/minibatch/dynamic
microbatch slicing。`move_micro_batch_without_teacher` 在 Student forward 前仅移动
其余输入，Teacher microbatch 仍为 CPU dense。当前 Student forward 完成后才调用
`teacher_action_logprobs`，将已经切好的 `[B_mu,T,V]` 搬到 Student device。
随后按相同 prediction rows 和 action mask 取 `[N_action_mu,V]`，与 Student shape
逐项检查。没有 batch 外部索引表，没有完整 Teacher train batch 的提前 H2D。

## 7. Full Student loss 的真实调用链

```text
_update_actor -> ActorRolloutRefWorker.update_actor
 -> TrainingWorker.train_mini_batch -> train_batch
 -> FSDPEngine.train_batch -> forward_backward_batch -> forward_step
 -> prepare_model_inputs -> patched model -> FusedLinearForPPO -> patched Liger
 -> prepare_model_outputs（保留 full_log_probs autograd）
 -> distillation_ppo_loss -> distillation_loss
 -> registry: compute_coverage_adaptive_full_vocab
 -> response_from_nested + action mask + teacher_action_logprobs
 -> 原有 coverage_adaptive_full_vocab
 -> native agg_loss / 全局 token denominator
 -> 原生 backward / FSDP sync / grad clip / optimizer_step / scheduler
```

Full registry 独立于 Top-K，不读取 teacher_ids。Teacher、Coverage/router 和 alpha
保持 stop-gradient；Student RKL/REC 概率保留梯度。Coefficient、温度、epsilon、
exact nucleus、residual bucket、FP32 和原有高/中/低 Coverage 行为保持原实现。

## 8. Causal alignment 验证

vLLM score[j] 预测 token[j]，上下文为 tokens[:j]。解码后存到 row[j-1]。
Student 对 prompt 长度 P、response 位置 r 使用 `response_from_nested` 取 P-1+r；
Teacher 选取完全相同的 row。首 response 为 P-1，末 response/EOS 为 L-2。
Tool observation 在上下文中保留，在 response_mask=0 的位置不计算蒸馏。

测试用可区分的 prefix-dependent 分布、递增位置编码、不同 prompt/response 长度、
EOS、mask0 observation 和 sample_id 检查 token、vocab、sample、padding 四类对齐。
乱序完成的请求使用唯一 identity 返回，不依赖完成顺序。

## 9. 测试结果与资源边界

本次迁移第一轮完整 CPU 回归已得到 **124 passed, 21 skipped**；随后发现并修复了
真实 TQ 的 nested repacking，修复相关测试 **7 passed**。最终约束内复测记录见
[原始验证记录](full_vocab_opd_validation.txt)。

五类最小用例覆盖：短单 trajectory full decode；多 trajectory 原生 Manager/client
并发乱序；带 observation 的 action mask；多个 native dynamic microbatch 的样本与
Teacher 同步重排及后置传输；小 Llama 现有 Student operator/loss/backward/grad clip/
optimizer 更新，检查 Student 非零梯度和参数变化、Teacher 无 graph/梯度。

最终串行 CPU 回归为 **130 passed, 21 skipped, 6 warnings in 20.01s**。
21 个 skip 为 20 个 CUDA 用例和 1 个 opt-in Ray 用例。测试采用内存监测保护，
含系统现有进程的 total cgroup 内存峰值 **1560.8 MiB**，停止阈值 1850 MiB，
host hard limit 为 2147483648 bytes，未触发停止或 OOM。Ruff、AST/shell 语法、
`git diff --check`、Liger patch reverse-check 及 Student 文件 SHA256 一致性均通过。

新增的 B=1/B=3 低内存用例直接调用已安装 TransferQueue 0.1.11 的
`AsyncSimpleStorageManager._pack_field_values`，再执行生产 `_async_meta_to_realdata`；
只替换远程读取端，验证实际 nested 打包恢复为 dense CPU。该检查不启动 Ray，
因此不替代跨进程 transport 测试。

CPU 协议测试使用实际 Manager 和 LLMServerClient.generate，RPC endpoint 是进程内
替身，**没有执行真实 vLLM GPU scheduler**。CPU engine 用例使用真实小模型和现有
算子/engine preparation/loss，但不声称运行了 GPU FSDP collectives。

当前 AutoDL 无卡模式，用户要求测试总内存不超过 2GB。真实 Ray/TQ 集成之前完成了
一次执行，发现 nested packing；修复后的真实 Ray 复测未取得成功完成记录，后续在
该限制下保留为 opt-in。没有把上一版旧缓存的 Ray 通过记录当作本版结果。
CUDA kernel、real vLLM Teacher/FSDP2 V1 单步、GPU H2D 与 OOM/吞吐实测均未执行。
对应 opt-in 测试已更新，待 GPU 环境运行。

CPU uv 环境：Python 3.12.3、torch 2.13.0+cpu、patched Liger 0.8.4、transformers 4.57.1、
tensordict 0.10.0、TransferQueue 0.1.11、Ray 2.59。GPU production pin 包含 transformers
5.16.1/vLLM 0.32.0；本机未验证该 GPU 依赖组合。

## 10. 运行命令

模型、数据和资源变量详见 [运行指南](full_vocab_opd.md)。在准备好的 GPU 环境内：

```bash
cd /root/autodl-tmp/verl-full-vocab-opd
uv sync --extra fsdp --extra vllm
source .venv/bin/activate
bash patches/liger/install.sh
# 导出运行指南表中的必填 OPD_* 参数后：
bash examples/on_policy_distillation_trainer/run_full_vocab_adaptive_opd.sh
```

CPU 回归（本次实际使用的测试清单；2GB 机器应在内存监测下串行执行）：

```bash
OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  -p pytest_asyncio.plugin \
  tests/full_vocab_opd \
  tests/utils/test_experimental_torch_functional_on_cpu.py \
  tests/utils/test_transferqueue_utils_on_cpu.py \
  tests/workers/test_distillation_topk_symmetry_on_cpu.py \
  tests/workers/test_engine_forward_step_detach_on_cpu.py \
  tests/trainer/ppo/v1/test_trainer_base_on_cpu.py \
  tests/trainer/ppo/v1/test_checkpoint_callback_on_cpu.py \
  tests/trainer/ppo/v1/test_rollout_profiling_on_cpu.py \
  tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py \
  tests/experimental/agent_loop/test_agent_loop_extra_fields_schema_on_cpu.py
```

以下是需要额外资源的 opt-in 命令，本次 2GB 无卡环境不宣称其通过：

```bash
OPD_RUN_RAY_TESTS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -s tests/full_vocab_opd/test_transfer_queue_integration.py

# GPU 环境，先导出同一 YAML 的 OPD_* 参数；smoke 自行生成短 parquet：
OPD_RUN_GPU_SMOKE=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -s tests/full_vocab_opd/test_v1_gpu_smoke.py
```
