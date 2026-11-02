我认为，现在可以把我们的研究整理成一套相当完整的 OPD 方法了。为了保证学术表述严谨，我会把内容分成四层：
研究背景： 为什么 OPD 通常使用 Reverse KL？它有什么优势和局限？
研究动机： 为什么我们认为 RKL 需要 Student 对 Teacher modes 具有足够的 coverage？
算法设计： 如何通过 Teacher Top-\(p\)、Student Coverage 和 Residual-aware FKL 实现动态 Recovery–Refinement？
与现有研究的关系： 我们与 EOPD、DrOPD、TIDE、ReTaCo 的区别是什么？
先给出整个研究最核心的一句话：
\[ \boxed{ \text{先利用 FKL 恢复严重欠覆盖的 Teacher-supported modes，} \quad \text{再利用 RKL 在已有 support 上进行 Mode-seeking Refinement。} } \]
我们不是希望 Student 永远保持高熵，也不是无条件地让 Student 覆盖 Teacher 的所有低概率尾部，而是希望解决 RKL 在严重欠覆盖情况下的恢复困难，让 RKL 更有效地发挥它的 Refinement 优势。
一、研究背景：OPD 为什么使用 Reverse KL？
1. On-Policy Distillation 的基本框架
On-Policy Distillation（OPD）的基本思想是：
Student 自己生成 trajectory，然后 Teacher 在 Student 实际访问的上下文中提供 token-level 的分布监督。
对于当前一个 Student-generated prefix：
\[ s_t=(x,y_{<t}), \]
Teacher 和 Student 分别产生：
\[ p_i=\pi_T(v_i\mid s_t) \]
\[ q_i=\pi_\theta(v_i\mid s_t). \]
其中：
\(p\)：Teacher 的完整 vocabulary probability distribution。
\(q\)：Student 的完整 vocabulary probability distribution。
OPD 希望通过 Teacher 的分布监督，让 Student 在自己实际访问的状态上逐渐接近 Teacher。
传统 OPD 经常采用 Reverse KL：
\[ \boxed{ L_t^{RKL} = D_{\mathrm{KL}}(q\|p) = \sum_{i=1}^{V} q_i\log\frac{q_i}{p_i} } \]
为什么选择它？
因为 RKL 具有一个非常重要的特点：它由 Student 分布 \(q\) 加权。
具体来说：
\[ \boxed{ \begin{aligned} &\text{Reverse KL 由 Student 分布 }q\text{ 加权；}\\ &\text{它强烈惩罚 Student 在 Teacher 低密度区域放质量；}\\ &\text{却相对容忍 Student 没有覆盖 Teacher 的部分高密度区域。} \end{aligned} } \]
这里需要特别注意最后一句的准确含义。
假设：
\[ p_i\gg0,\qquad q_i\rightarrow0. \]
那么：
\[ q_i\log\frac{q_i}{p_i} \rightarrow0. \]
也就是说，如果 Student 几乎不给某个 Teacher-supported token 概率，那么这个 token 对 RKL 的直接贡献反而很小。
反过来，如果：
\[ q_i\gg p_i, \]
Student 在 Teacher 不太认可的区域放了大量概率质量，那么：
\[ q_i\log\frac{q_i}{p_i} \]
就可能非常大。
这就是 RKL 的分布加权不对称性。
它也是经典 Variational Inference 中 Reverse KL 的 zero-forcing 特性的来源：倾向于避免在目标分布低密度的区域放置概率质量，而可能忽略目标分布中的部分模式。
mit.edu+1
2. RKL 为什么有 Mode-seeking 特性？
Mode-seeking 是对 RKL 在特定优化条件下行为的一种经典描述。
假设 Teacher 有多个明显的高概率区域，而 Student 的模型表达能力或者优化条件有限。
RKL 更关注：
\[ q_i>0 \]
的区域。
因此，Student 可以优先拟合自己已经具有 support 的某些 Teacher modes，而不必同等程度地覆盖 Teacher 的所有 modes。
这可能产生：
Student 集中于少数 modes；
Student 忽略部分 Teacher-supported alternatives；
Student 的分布相对集中；
生成多样性可能下降。
但是必须严谨地强调：
Mode-seeking 并不意味着 RKL 一定只选择 Teacher 的最高峰，也不意味着它一定只学习一个 mode。
它描述的是一种倾向，而不是无条件成立的数学定理。
特别是在完整 vocabulary、Student 表达能力足够、优化充分的情况下，两个方向的 KL 都可以在：
\[ q^*=p \]
处达到全局最优。
所以，我们研究的不是 RKL 在任何情况下都不好，而是它在实际 OPD 训练中可能存在的特定优化困难。
ICML 2026 的 EOPD 研究也观察到，RKL 在 Teacher 高熵位置可能降低生成多样性，并导致学习信号不稳定。
proceedings.mlr.press
二、Forward KL 的权威解析：为什么它能够解决这个问题？
Forward KL 定义为：
\[ \boxed{ L_t^{FKL} = D_{\mathrm{KL}}(p\|q) = \sum_{i=1}^{V} p_i\log\frac{p_i}{q_i} } \]
与 RKL 最重要的区别是：
\[ \boxed{ \text{Forward KL 由 Teacher 分布 }p\text{ 加权。} } \]
它关注的是：
Teacher 认为重要的概率质量，Student 有没有充分覆盖？
假设：
\[ p_i=0.20,\qquad q_i=10^{-6}. \]
Forward KL 对应的单项是：
\[ 0.20\log\frac{0.20}{10^{-6}}. \]
这个值很大。
当：
\[ q_i\rightarrow0,\qquad p_i>0 \]
时：
\[ p_i\log\frac{p_i}{q_i} \rightarrow+\infty. \]
所以，Forward KL 非常不希望 Student 忽略 Teacher 有概率质量的区域。
这就是经典的 zero-avoiding / mode-covering 特性。
1. 最重要的是 Logit Gradient
假设 Student 的 logits 为：
\[ z=[z_1,\ldots,z_V] \]
且：
\[ q_i=\operatorname{Softmax}(z)_i. \]
Forward KL 对 Student Logit 的梯度为：
\[ \boxed{ \frac{\partial L^{FKL}}{\partial z_i} = q_i-p_i } \]
因此，当：
\[ q_i\rightarrow0 \]
时：
\[ \boxed{ \frac{\partial L^{FKL}}{\partial z_i} \rightarrow-p_i } \]
只要 Teacher 的概率不为零，就仍然存在明确的恢复信号。
Gradient Descent 会提高对应的 Student logit。
这就是我们选择 FKL 的根本原因：
它能够针对 Teacher 高概率、Student 极低概率的 token，提供不会随着 Student probability 趋近于零而消失的直接恢复梯度。
2. Forward KL 和 Reverse KL 的梯度对比
RKL：
\[ \boxed{ \frac{\partial L^{RKL}}{\partial z_i} = q_i \left( \log\frac{q_i}{p_i} - D_{\mathrm{KL}}(q\|p) \right) } \]
FKL：
\[ \boxed{ \frac{\partial L^{FKL}}{\partial z_i} = q_i-p_i } \]
两种 KL 的核心梯度差异
Reverse KL
\[ g_i^R=q_i\left(\log\frac{q_i}{p_i}-D_R\right) \]
Student probability 乘子使严重欠覆盖 token 的直接恢复信号可能非常弱。
Forward KL
\[ g_i^F=q_i-p_i \]
Teacher probability 直接提供恢复信号，不会因为 Student probability 很低而消失。
因此：
\[ \boxed{ \begin{aligned} RKL &: \text{擅长惩罚 Student 在 Teacher 低密度区域放质量}\\ FKL &: \text{擅长惩罚 Student 对 Teacher-supported 区域的欠覆盖} \end{aligned} } \]
这就是我们算法的理论起点。
另外，必须补充一个学术边界：Mode-seeking / Mode-covering 是经典受限分布逼近下的行为解释，并不是对所有离散 LLM 蒸馏过程的无条件定理。2025 年 COLING 的《Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models》就对直接将连续分布中的 Mode-seeking 结论套用到离散 LLM KD 提出了质疑。
aclanthology.org
三、我们的研究出发点：RKL 的两个相关但不同的问题
现在进入我们真正想研究的内容。
我认为我们的研究动机应该分成两个层次。
1. 第一层：单个 Teacher mode 是否能够被 Student 学起来？
考虑某个固定的 Student-generated state：
\[ s_t. \]
Teacher 认为某个 token 非常重要：
\[ p_i=0.20. \]
但 Student：
\[ q_i=10^{-6}. \]
我们已经知道：
\[ g_i^{RKL} = q_i \left( \log\frac{q_i}{p_i}-D_R \right). \]
当：
\[ q_i\rightarrow0 \]
时：
\[ g_i^{RKL}\rightarrow0. \]
这就是我们所说的：
\[ \boxed{\text{RKL Gradient Starvation}} \]
也就是说：
Teacher 明明认为这个 token 很重要，但 Student 几乎没有给它 probability support，RKL 对它的直接恢复信号却非常弱。
这时我们希望通过 FKL：
\[ g_i^{FKL}=q_i-p_i \]
提供额外的恢复信号。
因此，第一个问题是：
\[ \boxed{ \text{Can Student recover an individual under-covered Teacher mode?} } \]
2. 第二层：Student 是否具有足够丰富的 Teacher-relative mode coverage？
这是我们进一步提出的研究动机。
我们不只是希望 Student 能够学习某一个 Teacher 高概率 token。
我们希望 Student 对 Teacher 的多个重要 alternatives 都具有一定的 probability support。
为什么？
因为我们的目标最终还是希望使用 RKL 进行 refinement。
假设 Teacher：
\[ p=[0.30,0.25,0.20,0.15,0.10]. \]
Student：
\[ q=[0.90,0.06,0.02,0.015,0.005]. \]
Student 对第一个 token 有很强的 support，但对其他 Teacher-supported alternatives 的 support 很弱。
这意味着：
第一个 mode 已经能够被 RKL 优化；
其他 modes 可能存在严重的 gradient starvation；
Student 的有效 support 可能过于集中。
我们希望先恢复多个重要 alternatives 的 support，再让 RKL 进行 refinement。
所以第二个问题是：
\[ \boxed{ \text{Does Student have sufficient support across important Teacher alternatives?} } \]
这两个问题的关系是：
两个层次的研究动机
Token-level learnability
单个 Teacher-supported token 是否能够被 Student 学起来？
关注：
\[ q_i/p_i \]
以及对应的 Logit Gradient。
Distribution-level coverage
Teacher 的核心 probability mass 中，有多少已经在 Student 中获得足够 support？
关注：
\[ C_t \]
Optimization strategy
Coverage 不足：FKL Recovery。
Coverage 充足：RKL Refinement。
这里有一个重要的学术边界：
我们提出的是一个研究假设：充分的 Teacher-relative coverage 有利于 RKL 在多个已有 support 的 alternatives 上进行有效 refinement。
目前不能声称已经有定理证明 RKL 必须先满足某个 coverage 阈值才能有效优化。
另外，我们的 Coverage 是 token-distribution-level 的概率质量指标，并不直接等于完整 sequence-level 的 mode 数量。
四、我们的算法：Coverage-Adaptive Recovery–Refinement OPD
现在正式介绍算法。
我们的核心设计是：
\[ \boxed{ \text{Teacher Top-}p \rightarrow \text{Student Coverage} \rightarrow \text{Adaptive RKL/FKL} } \]
Step 1. 获取 Teacher Top-\(p\) Tokens
在 Student rollout 到达的 state：
\[ s_t \]
上，分别获取：
\[ p=\pi_T(\cdot|s_t) \]
\[ q=\pi_\theta(\cdot|s_t). \]
这里我们使用完整 vocabulary probability distribution，而不是只使用 sampled token。
按照 Teacher probability 从高到低排序，选取累计概率质量达到 \(\rho\) 的最小集合：
\[ \boxed{ \mathcal N_t^\rho = \operatorname{TopP}(p,\rho) } \]
第一版实验可以设置：
\[ \rho=0.8. \]
这里的目的不是让 Student 学习整个 vocabulary 的所有长尾 token，而是首先关注 Teacher 最重要的 probability mass。
Step 2. 判断 Student 对 Teacher Tokens 的相对 Support
对于 Teacher nucleus 中的每个 token：
\[ i\in\mathcal N_t^\rho \]
计算：
\[ \boxed{ r_i=\frac{q_i}{p_i} } \]
设定一个相对 support 阈值：
\[ \tau. \]
如果：
\[ \frac{q_i}{p_i}\geq\tau \]
我们认为该 token 已经获得足够的 Student support。
如果：
\[ \frac{q_i}{p_i}<\tau \]
则将它判定为 deficit token。
于是：
\[ \boxed{ \mathcal D_t = \left\{ i\in\mathcal N_t^\rho: \frac{q_i}{p_i}<\tau \right\} } \]
这就是我们的 Deficit Set。
Step 3. 计算 Teacher-relative Coverage
定义：
\[ \boxed{ C_t = \frac{ \sum_{i\in\mathcal N_t^\rho} p_i\mathbf1[q_i/p_i\geq\tau] }{ \sum_{i\in\mathcal N_t^\rho}p_i } } \]
它表示：
Teacher nucleus 中，有多少比例的 probability mass 已经获得足够的 Student-relative support？
也可以写成：
\[ \boxed{ C_t = 1- \frac{ \sum_{i\in\mathcal D_t}p_i }{ \sum_{i\in\mathcal N_t^\rho}p_i } } \]
这就是我们的 Router。
注意：Coverage 越高，说明 Student 对 Teacher 核心 probability mass 的覆盖越充分，但不意味着 Student 的 entropy 必然越高。
Step 4. 构造 Recovery Loss
这里我们不采用简单的 D-only FKL，因为那样可能推动：
\[ Q_D\rightarrow1. \]
也不采用重新归一化 Top-\(p\) 的 FKL，因为那样会错误地要求被选中的 tokens 占据全部概率质量。
我们采用：
\[ \boxed{ L_t^{REC} = \underbrace{ \sum_{i\in\mathcal D_t} p_i\log\frac{p_i}{q_i} }_{\text{Deficit-token FKL}} + \underbrace{ (1-m_D) \log\frac{1-m_D}{1-Q_D} }_{\text{Residual-category FKL}} } \]
其中：
\[ m_D=\sum_{i\in\mathcal D_t}p_i \]
\[ Q_D=\sum_{i\in\mathcal D_t}q_i. \]
其本质是将 Deficit Set 以外的所有 tokens 合并成一个 Residual Category，然后在聚合分布上计算 Forward KL。
因此：
Deficit tokens：逐 Token 恢复 Teacher probability。
Residual category：保证剩余 Vocabulary 保留合理的总概率质量。
不要求剩余 Vocabulary 内部的概率分布通过 FKL 完全匹配。
最重要的梯度性质：
\[ \boxed{ \frac{\partial L_t^{REC}}{\partial z_i} = q_i-p_i,\qquad i\in\mathcal D_t } \]
这保证了严重欠覆盖的 Teacher tokens 能够获得明确的恢复梯度。
这个 residual 设计与 ReTaCo 保留剩余 Vocabulary 概率质量的思想相通，但我们的恢复集合是根据 Student-relative coverage 筛选出的 deficit set。
arxiv.org
Step 5. 构造 Coverage-Adaptive Mixing Weights
现在，我们让 Coverage 决定 FKL 和 RKL 的比例。
设定：
\[ \delta_C=0.8. \]
定义：
\[ \boxed{ \alpha_F(C_t) = \alpha_{\max} \operatorname{clip} \left( \frac{\delta_C-C_t}{\delta_C}, 0,1 \right) } \]
以及：
\[ \boxed{ \alpha_R(C_t)=1-\alpha_F(C_t) } \]
第一版实验可以尝试：
\[ \alpha_{\max}=0.8. \]
于是：
Coverage
RKL
Recovery FKL
策略
0.9
100%
0%
Full RKL
0.8
100%
0%
Full RKL
0.6
80%
20%
Hybrid
0.4
60%
40%
Hybrid
0.2
40%
60%
Recovery-heavy
0.0
20%
80%
Recovery-heavy
这里：
\[ C_t\geq0.8 \]
直接使用 Full RKL。
而：
\[ C_t<0.8 \]
使用 Hybrid。
为什么保留最低 20% 的 RKL？
因为我们希望在 Recovery 阶段仍然保留 RKL 对 Student-excess 的抑制和 refinement 能力。
Step 6. 最终 Loss
现在，所有内容汇总为：
\[ \boxed{ L_t = \alpha_R(C_t)L_t^{RKL} + \alpha_F(C_t)L_t^{REC} } \]
其中：
\[ L_t^{RKL} = \sum_{i=1}^{V} q_i\log\frac{q_i}{p_i}. \]
这就是我们完整的 Token-level Objective。
注意：
RKL 始终作用于完整 vocabulary。
REC 只针对 deficit tokens 进行逐 Token FKL，同时保留 residual mass。
Router、Deficit Mask 和 mixing weights 使用 stop-gradient。
所有 Loss 共享同一个 Student Softmax distribution。
另外，\(\alpha_R,\alpha_F\) 是 Loss 权重，不代表实际参数梯度贡献的精确比例。实现时需要监控两路梯度范数。
五、与现有研究的关系：我们的研究究竟处于什么位置？
这是写论文时非常重要的一部分。
我认为目前最值得对照的是下面四项研究。
[Entropy-Aware On-Policy Distillation of Language Models | Papers with Code](https://images.openai.com/static-rsc-4/5nr0TaeS9tRkusEOzMq14A_grCJJh6mLiFoZ5hLIOVqpYOBMokjGUAxtzJ-ZdKoFrNLvY1ZMNLL_fnR4kFfYaTmVgCB4mi1h-3khlThHDKFslnYm7ou9bQ60EdfEeTh5pBECejTj5JOJHLEUMejLzA-_oPW7YFLNlqOvkvlCjPE?purpose=inline)
1. EOPD — Entropy-Aware On-Policy Distillation
ICML 2026
核心思路：根据 Teacher entropy 判断不确定性较高的位置，在 RKL 之外增加 FKL。
主要目标：保留 Teacher 的 plausible alternatives，平衡 mode-seeking precision 与 mode-covering robustness。
[论文：PMLR 2026](https://proceedings.mlr.press/v306/jin26e.html?utm_source=chatgpt.com)
[On-Policy Distillation：从分布视角理解 SFT、RL 与 OPD | ZijieGuo 的技术博客](https://images.openai.com/static-rsc-4/soSWe9-EokBEhAWrqFPIL5a-FhMMm3jX0oAv5NKKR3BdQxObma5tsoEhbtNzv6dntRxmGzE9mKEfBAhlWryH_GPPK8NR1yEtLrFUMeyqYaHnmxXEmhksbUAo66QfcNTegoqV7TOxXLy6m5pZg8fscyObReEAqg7CMuDxjiUYLTE?purpose=inline)
2. DrOPD — Divergence-Routed On-Policy Distillation
2026 arXiv
核心思路：RKL 在 Teacher-supported、Student-low-probability tokens 上可能提供很弱的纠正信号，因此保留 RKL，并在高 JS divergence 的位置增加 FKL。
这篇与你的 Gradient Starvation 动机非常接近。
[论文：arXiv](https://arxiv.org/abs/2610.19572?utm_source=chatgpt.com)
[Ksenia_TuringPost (@TheTuringPost) on X](https://images.openai.com/static-rsc-4/E9Ljq0QPzhoDeWTPypN6MZGt7seuV4b2CsOjAtCkfyu4fDci-5TlVhKoBcnxltf9lbrt2Wkz5zBJpjDk_3RlS1gWyWv8LrPtu1nJ57VQ9Z-xWVIAL70EOdF0UtLYTWMxJEdaW_D2VPHuUQttqqQtUPS9gRMn_x2-M0c-owCMUs0?purpose=inline)
3. TIDE — Token-level Independent Deficit-Excess Correction
2026 arXiv
核心思路：区分 Student-excess 和 Student-deficit。使用 Hellinger shaping 处理严重 excess，并通过 Teacher Top-\(K\) injection 恢复 deficit。
与我们对 deficit token 的分析有直接关联。
[论文：arXiv](https://arxiv.org/abs/2608.09836?utm_source=chatgpt.com)
[Shuai Kyle Zheng (@bittnt) on X](https://images.openai.com/static-rsc-4/D09Scv589d2U8ZRe7mchutErgoBDqYslwuc6d7T_uiVfmbbMWTrYmQeP7Q4PkJyxhfSCq6b6N5LRrYcsqbEZPDiy0sNrXnKfoTfsmDbnEACVC_MjNYGnc75JwF5bRtEEEYx_Qtt9POlDnpwKuztJ1XMqwlFNndM6L7NbmRGwYsY?purpose=inline)
4. ReTaCo — Residual-Target Control for On-Policy Distillation
2026 arXiv
核心思路：Top-\(K\) 之外的 probability mass 不应该直接丢弃，而应该通过 residual symbol 显式保留。
与我们 Recovery Loss 的 residual category 设计高度相关。
[论文：arXiv](https://arxiv.org/abs/2609.39275?utm_source=chatgpt.com)
这四篇研究分别给我们提供了不同的启发：
研究
对我们的启发
EOPD
FKL 和 RKL 可以互补
DrOPD
RKL 欠覆盖问题需要额外 FKL
TIDE
可以直接针对 deficit tokens 进行恢复
ReTaCo
必须合理处理被选择集合以外的 residual mass
我们目前的差异化设计在于：
\[ \boxed{ \text{Student-relative Top-}p\text{ Coverage Routing} } \]
不是根据 Teacher entropy 判断是否需要 FKL，也不是根据 JS divergence 选择需要干预的位置，而是直接查看：
Teacher 的重要 probability mass，有多少已经获得 Student 的相对 support？
随后：
\[ \boxed{ \text{Coverage} \rightarrow \text{Deficit Set} \rightarrow \text{Residual-aware FKL} \rightarrow \text{Adaptive RKL/FKL} } \]
这是一条相对完整的研究路线。
不过，必须客观地说：各个组成思想都已有相关研究，能否构成足够的新颖性，需要进一步与 DrOPD、TIDE、ReTaCo 的具体算法和实验逐项比较，不能仅凭组合方式就宣称已经证明了方法创新。
六、我们的研究应该如何设计实验来证明？
我建议至少设置下面六组实验：
实验
方法
验证目标
A
Full RKL
标准 OPD baseline
B
Full FKL
检查全面 mode-covering 的效果
C
Fixed 50/50 RKL + FKL
检查固定混合是否足够
D
EOPD / DrOPD
对比已有自适应混合方法
E
RKL + D-only FKL
验证 residual correction 是否必要
F
Ours: Coverage-adaptive RKL + REC
验证完整算法
我尤其建议关注以下指标：
Coverage recovery： 训练后 Student 对 Teacher nucleus 的覆盖是否提升？
Deficit ratio： 严重欠覆盖的 Teacher probability mass 是否下降？
Student entropy： 是否出现不必要的 entropy collapse？
Pass@1 / Pass@K： 单次能力和多次采样能力是否同时改善？
Downstream accuracy： 最终任务能力是否真正提高？
Gradient statistics： Recovery 和 RKL 两路梯度是否稳定，是否存在严重干涉？
特别需要验证：
\[ \text{Coverage}\uparrow \]
是否真的能够带来：
\[ \text{RKL refinement effectiveness}\uparrow. \]
这才是我们研究假设最关键的实验验证。
七、最终研究总结：可以直接作为论文 Introduction 的核心逻辑
下面是我建议你最终采用的学术表述。
Coverage-Adaptive Recovery–Refinement On-Policy Distillation
研究背景
On-Policy Distillation 通过在 Student-generated states 上提供 Teacher 的 token-level distribution supervision，实现对 Student 当前行为分布的直接优化。传统 OPD 通常采用 Reverse KL，以发挥其对 Student-excess probability 的抑制作用及其 mode-seeking 特性。
然而，RKL 由 Student distribution 加权，对于 Teacher-supported 但 Student 严重欠覆盖的 tokens，其 Logit Gradient 可能趋近于零，使得部分重要 Teacher alternatives 难以被有效恢复。
研究动机
我们认为，RKL 的有效 refinement 不仅涉及单个 Teacher mode 的可学习性，还涉及 Student 对多个重要 Teacher alternatives 的整体 support。
因此，我们希望首先诊断 Student 对 Teacher 核心 probability mass 的覆盖程度，并在严重欠覆盖时提供额外的恢复信号，避免 RKL 过早地局限于狭窄的有效 support。
核心方法
我们提出基于 Teacher Top-\(p\) nucleus 的 Student-relative Coverage Metric，通过比较 Teacher 与 Student 对应 tokens 的概率比值，识别严重欠覆盖的 Teacher alternatives，并计算当前 state 的 coverage。
在此基础上，构建 Deficit-selective Residual-aware Forward KL Recovery Loss：对严重欠覆盖的 tokens 提供直接的 FKL 恢复梯度，同时将其余 vocabulary 聚合为 residual category，以保留剩余 probability mass。
最后，通过 Coverage-Adaptive Loss Weighting 动态调整 RKL 与 Recovery FKL 的比例：
\[ \boxed{ L_t= \alpha_R(C_t)L_t^{RKL} + \alpha_F(C_t)L_t^{REC} } \]
预期效果
当 Student coverage 不足时，增加 FKL recovery，恢复缺失的 Teacher-supported alternatives，同时保留 RKL 的 refinement 作用；随着 coverage 提升，逐渐降低 FKL 权重，最终过渡到 Full RKL，使 Student 在已经获得足够 support 的区域内进行 mode-seeking refinement。
最终研究目标
提高 Reverse KL 在 On-Policy Distillation 中的优化有效性，使其既能够恢复重要的 Teacher-supported alternatives，又能够充分发挥自身的 refinement 优势。
最后，我想帮你确定三个最重要的理论结论
第一：我们不是单纯研究 Forward KL 和 Reverse KL 的融合。
我们真正研究的是：
\[ \boxed{ \text{When should RKL receive additional recovery supervision?} } \]
也就是：什么时候 Student 的 Teacher-relative coverage 已经足够，能够主要依赖 RKL？
第二：我们引入 FKL，不是为了无条件提高 Student entropy。
而是为了：
\[ \boxed{ \text{Recover severely under-covered Teacher-supported alternatives.} } \]
它恢复的是 Teacher 认为重要、Student 却几乎没有 support 的 alternatives。
第三：我们的最终目标仍然是 RKL。
\[ \boxed{ \text{FKL Recovery} \rightarrow \text{Sufficient Student Support} \rightarrow \text{RKL Refinement} } \]
这就是整个算法的研究故事。
我认为这条研究路线最值得强调的地方，是我们把一个原本比较抽象的问题——Reverse KL 的 Mode-seeking 是否会因为 Student 对 Teacher modes 的欠覆盖而难以有效发挥？——转化成了一个可以直接在完整 vocabulary probability distribution 上计算、诊断和优化的问题。
这才是我们应该围绕展开论文的核心。