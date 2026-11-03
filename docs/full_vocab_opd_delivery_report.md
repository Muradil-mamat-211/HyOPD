# Full-Vocab OPD：vLLM Teacher 迁移与 correctness 交付报告

当前工作目录：`/root/autodl-tmp/HyOPD-full-vocab-publish`，分支
`full-vocab-vllm-teacher`。原实现目录为 `/root/autodl-tmp/verl-full-vocab-opd`。veRL baseline 为
`3fdf2b548882a8aeccf0842c2150d9c9f1790a55`，保留既有 Student 实现后完成此次迁移。
实现与验证阶段未 commit/push；随后按用户要求发布至 `Muradil-mamat-211/HyOPD`
的 `full-vocab-vllm-teacher` 分支。保留 `verl-latest` 和原 `HyOPD` 工作区的既有改动。
迁移前文件备份：`/root/autodl-tmp/verl-full-vocab-opd-before-vllm.tar.gz`。

## 本轮 correctness 修复

本轮只调整 formal mixing schedule、route metric 边界和 Teacher transfer 顺序。

| 文件 | 本轮修改 |
|---|---|
| `verl/trainer/distillation/coverage_adaptive_loss.py` | 配置化 alpha_max；inclusive Full-RKL route；其余 Coverage/RKL/REC 与 stop-gradient 保留 |
| `verl/workers/config/distillation.py` | 新增 `alpha_max=0.8` 并验证 finite / `[0,1]` |
| `verl/trainer/config/distillation/distillation.yaml` | 显式 alpha_max 默认值及边界注释 |
| `examples/on_policy_distillation_trainer/full_vocab_adaptive_opd.yaml` | 显式 `alpha_max: 0.8` |
| `verl/trainer/distillation/full_vocab_cache.py` | 当前 microbatch 在 CPU 选 action prediction rows，仅传 `[N_action,V]` FP32 |
| `tests/full_vocab_opd/test_coverage_loss.py` | 固定六个 Coverage 点、非默认 alpha_max、inclusive ratio、更新独立 reference，验证 Student gradient 与 Teacher detach |
| `tests/full_vocab_opd/test_dense_teacher_data.py` | spy 验证 Teacher `.to` 输入已为 action-only；action 数与 microbatch size 明确不同；保留重排/mask/EOS/zero-action 用例 |
| `tests/full_vocab_opd/test_yaml_config.py` | 验证 YAML 默认值和 alpha_max 非有限/越界拒绝 |
| `README.md`、`docs/full_vocab_opd.md` | 正式公式、action-only 传输和当前 CPU-only 验证边界 |
| 本报告、`full_vocab_opd_implementation_plan.md`、`full_vocab_opd_validation.txt` | 当前结果、最小命令和未验证项 |

公式为 `alpha_f = alpha_max * clip((coverage_threshold-C)/coverage_threshold, 0, 1)`，
`alpha_r = 1-alpha_f`。两个默认参数均为 0.8，C=0 保留 0.2 RKL；
C 恰为 0.8 时 Full-RKL route 为 True，alpha_r/f 为 1/0。

Teacher H2D 已变为 action-only：先在 CPU 按当前 TensorDict sample 顺序和
`P-1+r` 选出有效 prediction rows，再对 `[N_action,V]` 一次性执行
`.to(student_device, dtype=torch.float32)`。Teacher 仍 detached，无 teacher_ids。

本轮 **42 passed、0 failed**（指定六个 CPU 文件，12.48s），总 cgroup 内存峰值
**1665.8 MiB**；无 OOM。Ruff 和 `git diff --check` 通过。没有重装或改变环境。
本轮未运行 GPU smoke、真实 CUDA/vLLM、FSDP2 collective/NCCL、Ray E2E；
GPU peak memory / throughput 未验证，相关 opt-in 测试保留。
CPU correctness 覆盖范围内没有发现剩余 blocker；GPU 执行仍需后续实测。

## 1–2. 文件清单及具体修改

下表保留此前 Teacher-FSDP2 → vLLM 的迁移明细。本轮 correctness patch 以上节为准，
没有重新设计架构，也没有改动 Student 算子或 optimizer cadence。

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
| `verl/workers/engine/fsdp/transformer_impl.py` | 删除 Teacher 专属 CPU offload 分支；Student 当前 forward 前保留 Teacher CPU 字段，forward 后由 loss 在 CPU 选出并传输当前 action rows；保留 Student full output/autograd 处理 |
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

此前 Teacher backend 迁移阶段，上述 Student 数学/算子文件保持哈希一致。
本轮仅在 `coverage_adaptive_loss.py` 修正 README 的 mixing schedule 和 inclusive route；
`torch_functional.py`、Liger patch、模型输出补丁和 GPU smoke 文件仍未改动。

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
`teacher_action_logprobs`，先在 CPU 对已经切好的 `[B_mu,T,V]` 按相同 prediction rows
和 action mask 取 `[N_action_mu,V]`，与 Student shape 逐项检查，再一次传到 Student
device 并指定 FP32。没有 batch 外部索引表，也不传输整个 Teacher microbatch。

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
exact nucleus、residual bucket、FP32 保留，mixing schedule 按上节正式公式修正。

## 8. Causal alignment 验证

vLLM score[j] 预测 token[j]，上下文为 tokens[:j]。解码后存到 row[j-1]。
Student 对 prompt 长度 P、response 位置 r 使用 `response_from_nested` 取 P-1+r；
Teacher 选取完全相同的 row。首 response 为 P-1，末 response/EOS 为 L-2。
Tool observation 在上下文中保留，在 response_mask=0 的位置不计算蒸馏。

测试用可区分的 prefix-dependent 分布、递增位置编码、不同 prompt/response 长度、
EOS、mask0 observation 和 sample_id 检查 token、vocab、sample、padding 四类对齐。
乱序完成的请求使用唯一 identity 返回，不依赖完成顺序。

## 9. 测试结果与资源边界

当前 patch 的结果为 **42 passed、0 failed，6 warnings，12.48s**。
只运行下面六个 CPU-safe 文件；未运行全仓库 pytest、CUDA kernel 或真实 Ray transport。
本轮低内存监测的 cgroup 峰值 1665.8 MiB，停止阈值 1850 MiB，host hard limit
2147483648 bytes；未触发停止或 OOM。原始输出见 [验证记录](full_vocab_opd_validation.txt)。

覆盖默认六个 Coverage 点及自定义 alpha_max、边界 route ratio、完整 nucleus/deficit/
RKL/residual REC/加权 loss、Student finite gradient、Teacher detach、empty deficit 和
extreme probabilities。数据用例覆盖首末 response/EOS、variable length、tool mask、
padding、zero action、dynamic reorder 以及 action-only transfer。

小 Llama 用例在 CPU 运行现有 Student operator/engine preparation/loss/backward/
grad clip/SGD update；没有运行 FSDP2 distributed 初始化或 collective。Manager/client
用例的 RPC endpoint 为进程内替身，没有运行真实 vLLM engine/scheduler。
TQ 用例只调用已安装包的 packer 和生产 read conversion，替换远程 I/O，不启动 Ray。

CUDA/vLLM Teacher、GPU FSDP2 collective、NCCL、多 GPU Ray E2E、GPU peak memory
及 throughput 明确未验证。`test_v1_gpu_smoke.py` 原样保留为 opt-in，本轮不执行。
CPU 覆盖范围内没有发现剩余 correctness blocker。

此前 130 passed / 21 skipped 是迁移阶段的历史结果，不作为本轮更新代码的全套验收声明。
本轮复用现有 uv CPU 环境（torch 2.13.0+cpu、patched Liger 0.8.4、transformers 4.57.1），
未安装大型依赖或改变环境。GPU 依赖组合的通过状态不作声明。

## 10. 运行命令

模型、数据和资源变量详见 [运行指南](full_vocab_opd.md)。在准备好的 GPU 环境内：

```bash
cd /root/autodl-tmp/HyOPD-full-vocab-publish
uv sync --extra fsdp --extra vllm
source .venv/bin/activate
bash patches/liger/install.sh
# 导出运行指南表中的必填 OPD_* 参数后：
bash examples/on_policy_distillation_trainer/run_full_vocab_adaptive_opd.sh
```

本轮实际 CPU 测试清单（使用已有 CPU 环境，并由内存监测器串行运行）：

```bash
OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/full_vocab_opd/test_coverage_loss.py \
  tests/full_vocab_opd/test_dense_teacher_data.py \
  tests/full_vocab_opd/test_engine_chain.py \
  tests/full_vocab_opd/test_vllm_teacher.py \
  tests/full_vocab_opd/test_agent_teacher_flow.py \
  tests/full_vocab_opd/test_yaml_config.py
```

实际运行还断言了 `verl.__file__` 来自当前发布工作区，避免误测旧 editable source；
内存监测命令及日志见 validation 文件。静态检查仅覆盖本轮修改的 Python 文件。

以下是需要额外资源的 opt-in 命令，本次 2GB 无卡环境不宣称其通过：

```bash
OPD_RUN_RAY_TESTS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -s tests/full_vocab_opd/test_transfer_queue_integration.py

# GPU 环境，先导出同一 YAML 的 OPD_* 参数；smoke 自行生成短 parquet：
OPD_RUN_GPU_SMOKE=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q -s tests/full_vocab_opd/test_v1_gpu_smoke.py
```
