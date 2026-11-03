# Full-Vocab OPD 当前实现与迁移审计

此次在已经完成的 Student Full-Vocab 代码上，将 Teacher 从 FSDP2 forward-only
迁回原生 veRL vLLM inference。当前模式为 `full_vocab_vllm`；旧 Teacher-FSDP2
设计已被替换，不能再据此文档的早期版本部署。

## 源码与工作区

- veRL baseline：`3fdf2b548882a8aeccf0842c2150d9c9f1790a55`。
- 原实现工作区：`/root/autodl-tmp/verl-full-vocab-opd`。
- 当前发布与修复工作区：`/root/autodl-tmp/HyOPD-full-vocab-publish`，分支 `full-vocab-vllm-teacher`。
- Liger 0.8.4 base：`43293d84d760e402e4fd974b12410a1d47c31761`。
- Selected-logprob 审计来源：`8c7a5078318da1295dc3ca445f9d85e9e9e153f5`。
- vLLM API 审计版本：项目 pin `0.32.0`。
- 迁移前本地备份：`/root/autodl-tmp/verl-full-vocab-opd-before-vllm.tar.gz`。

## 完成的迁移

1. 恢复 MultiTeacherModelManager、原生 vLLM replica/resource pool、LLMServerClient。
2. AgentLoop 已完成的各 trajectory 立即并发发起 scoring，不等待 rollout batch/reward。
3. 请求 `prompt_logprobs=-1, flat_logprobs=true`，engine `max_logprobs=-1`，raw scores。
4. flat primitive arrays 直接解码完整 vocabulary，按 token ID 排列并 shift 一次；验证
   request identity、完整 collection、vocab coverage、finite values。
5. CPU temperature transform 后按容量 padding，Teacher `[B,T,V]` 使用原生 TQ；TQ
   读回时恢复 dense。Full mode 不生成 teacher_ids。
6. Teacher field 随 TensorDict 原生重排/切分；Student forward 后先在 CPU 选择有效
   action prediction rows，仅传输 `[N_action,V]` FP32。
7. 独立 Full registry 接回现有 Coverage/RKL/REC 算法，保留 backward/optimizer cadence。
8. 删除旧 Teacher-FSDP2 worker、driver scoring、预算/释放逻辑和配置。

Student 算子及模型 PPO 输出补丁保持未改动。本轮 correctness 修复只调整 mixing
schedule：`alpha_f=alpha_max*clip((coverage_threshold-C)/coverage_threshold,0,1)`，
`alpha_r=1-alpha_f`，默认 alpha_max=coverage_threshold=0.8。Full-RKL route 使用 `>=`。
Full-range Liger 继续使用既有 portable 路径，未新增 SM90 full-vocab 性能承诺。

## 验证边界

CPU 小模型、协议、异步乱序、action mask、dynamic microbatch、优化器更新和上游回归
记录见 [交付报告](full_vocab_opd_delivery_report.md) 及
[原始日志](full_vocab_opd_validation.txt)。
当前用户指定 AutoDL 无卡模式、测试内存总量不超过 2GB。实际 GPU vLLM/FSDP2 执行及
修复后的完整 Ray transport 用例保留为 opt-in，未取得通过记录时不计入已验证范围。

本轮仅六个指定 CPU correctness 文件：42 passed、0 failed；total cgroup 内存峰值
1665.8 MiB，未触发 OOM，Ruff / git diff --check 通过。没有安装或修改依赖。
未执行 CUDA/vLLM GPU、FSDP2 collective/NCCL、Ray E2E，未验证 GPU peak memory/throughput。
GPU smoke 原样保留为 opt-in。历史迁移阶段的 130 passed 不代表本轮全套测试已运行。

运行与配置见 [运行指南](full_vocab_opd.md)。实现验证完成后，按用户要求发布至 HyOPD 的独立分支。
