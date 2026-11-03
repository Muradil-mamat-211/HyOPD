# Full-Vocab OPD 当前实现与迁移审计

此次在已经完成的 Student Full-Vocab 代码上，将 Teacher 从 FSDP2 forward-only
迁回原生 veRL vLLM inference。当前模式为 `full_vocab_vllm`；旧 Teacher-FSDP2
设计已被替换，不能再据此文档的早期版本部署。

## 源码与工作区

- veRL baseline：`3fdf2b548882a8aeccf0842c2150d9c9f1790a55`。
- 独立工作区：`/root/autodl-tmp/verl-full-vocab-opd`。
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
6. Teacher field 随 TensorDict 原生重排/切分；Student forward 后才搬当前 Teacher microbatch。
7. 独立 Full registry 接回现有 Coverage/RKL/REC 算法，保留 backward/optimizer cadence。
8. 删除旧 Teacher-FSDP2 worker、driver scoring、预算/释放逻辑和配置。

Student 算子、模型 PPO 输出补丁和数学模块的 SHA256 与迁移前一致。
Full-range Liger 继续使用既有 portable 路径，未新增 SM90 full-vocab 性能承诺。

## 验证边界

CPU 小模型、协议、异步乱序、action mask、dynamic microbatch、优化器更新和上游回归
记录见 [交付报告](full_vocab_opd_delivery_report.md) 及
[原始日志](full_vocab_opd_validation.txt)。
当前用户指定 AutoDL 无卡模式、测试内存总量不超过 2GB。实际 GPU vLLM/FSDP2 执行及
修复后的完整 Ray transport 用例保留为 opt-in，未取得通过记录时不计入已验证范围。

最终 CPU 回归 130 passed、21 skipped；total cgroup 内存峰值 1560.8 MiB，
未触发 OOM。Student/operator/math 文件哈希一致，静态检查通过。

运行与配置见 [运行指南](full_vocab_opd.md)。实现验证完成后，按用户要求发布至 HyOPD 的独立分支。
