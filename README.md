# HyOPD: Hybrid On-Policy Distillation

> **Hybrid On-Policy Distillation via Coverage-Adaptive Recovery–Refinement**  
> 核心思想：先利用 FKL 恢复严重欠覆盖的 Teacher-supported modes，再利用 RKL 在已有 support 上进行 Mode-seeking Refinement。

---

## 一、研究背景 (Research Background)

### 1. On-Policy Distillation 的基本框架

On-Policy Distillation（OPD）的基本思想是：
Student 自主生成 trajectory，Teacher 在 Student 实际访问的上下文状态下提供 token-level 的分布监督。

对于当前 Student 生成的前缀状态：

$$
s_t = (x, y_{\lt t})
$$

Teacher 和 Student 分别产生词表 $V$ 上的完整概率分布：
- $p_i = \pi_T(v_i \mid s_t)$ ：Teacher 的完整词表概率分布
- $q_i = \pi_\theta(v_i \mid s_t)$ ：Student 的完整词表概率分布

传统 OPD 普遍采用 Reverse KL（RKL）：

$$
L_t^{RKL} = D_{\mathrm{KL}}(q \parallel p) = \sum_{i=1}^{V} q_i \log \frac{q_i}{p_i}
$$

---

### 2. 为什么选择 Reverse KL？优势与 Mode-seeking 特性

Reverse KL 具有一个关键特性：**它由 Student 分布 $q$ 加权**。

具体表现为：

$$
\begin{aligned}
&\text{Reverse KL 由 Student 分布 } q \text{ 加权；} \\
&\text{强烈惩罚 Student 在 Teacher 低密度区域分配质量（抑制 Student-excess）；} \\
&\text{相对容忍 Student 未完全覆盖 Teacher 的部分高密度区域。}
\end{aligned}
$$

#### Mode-seeking 特性
假设 Teacher 拥有多个明显的高概率模式（modes），而 Student 的模型表达能力或优化条件有限：
- RKL 更关注 $q_i \gt 0$ 的区域；
- Student 可以优先拟合自身已有 support 的某些 Teacher modes，而不必等额覆盖 Teacher 的全部模式；
- 这使得 Student 分布较为集中，生成精准，但可能伴随多样性下降的倾向。

---

### 3. Reverse KL 的局限：梯度饥饿 (Gradient Starvation)

当 Teacher 认为某个 token 非常重要 ($p_i \gg 0$) ，而 Student 几乎未分配概率质量 ($q_i \to 0$) 时：

$$
q_i \log \frac{q_i}{p_i} \longrightarrow 0 \quad (q_i \to 0)
$$

也就是说，Student 欠覆盖的 token 对 RKL 的直接贡献极小。

从 Logit Gradient 角度分析，设 Student logits 为 $z = [z_1, \dots, z_V]$ 且 $q_i = \text{Softmax}(z)_i$ ，RKL 对 Student logit 的梯度为：

$$
\frac{\partial L^{RKL}}{\partial z_i} = q_i \left( \log \frac{q_i}{p_i} - D_{\mathrm{KL}}(q \parallel p) \right)
$$

当 $q_i \to 0$ 时：

$$
\frac{\partial L^{RKL}}{\partial z_i} \longrightarrow 0 \quad (q_i \to 0)
$$

这就是 **RKL Gradient Starvation（梯度饥饿）**：
Teacher 明明赋予该 token 显著的概率质量，但由于 Student 几乎没有 support，RKL 针对该 token 的直接恢复梯度趋近于 0，使得严重欠覆盖的 Teacher alternatives 难以被有效恢复。

---

### 4. Forward KL 的互补特性与梯度对比

Forward KL 定义为：

$$
L_t^{FKL} = D_{\mathrm{KL}}(p \parallel q) = \sum_{i=1}^{V} p_i \log \frac{p_i}{q_i}
$$

与 RKL 的核心区别在于：**Forward KL 由 Teacher 分布 $p$ 加权**，具备经典的 **Zero-avoiding / Mode-covering** 特性。

当 $q_i \to 0, \, p_i \gt 0$ 时：

$$
p_i \log \frac{p_i}{q_i} \longrightarrow +\infty
$$

#### 梯度对比与恢复机制
FKL 对 Student Logit 的梯度为：

$$
\frac{\partial L^{FKL}}{\partial z_i} = q_i - p_i \longrightarrow -p_i \quad (q_i \to 0)
$$

只要 Teacher 的概率 $p_i \gt 0$ ，梯度下降就会直接增大对应的 Student logit，提供不会随 Student 概率衰减的明确恢复信号。

| 损失函数 | 权重分布 | 核心 Logit 梯度 $g_i = \frac{\partial L}{\partial z_i}$ | 优化特性 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| **Reverse KL** | Student $q$ 加权 | $q_i \left( \log \frac{q_i}{p_i} - D_{\mathrm{KL}} \right)$ | 强烈惩罚 Student 在 Teacher 低密度区的虚假质量（抑制 Excess） | **Refinement** (已有 support 上的精细优化) |
| **Forward KL** | Teacher $p$ 加权 | $q_i - p_i$ | 强烈惩罚 Student 对 Teacher 模式的欠覆盖（提供 $-p_i$ 恢复信号） | **Recovery** (严重欠覆盖模式的快速唤醒) |

---

## 二、算法设计 (Algorithm Design: Coverage-Adaptive Recovery–Refinement OPD)

### 算法核心架构

$$
\text{Teacher Top-}p \;\longrightarrow\; \text{Student Coverage} \;\longrightarrow\; \text{Deficit Set} \;\longrightarrow\; \text{Residual-aware FKL} \;\longrightarrow\; \text{Adaptive RKL/FKL}
$$

---

### Step 1. 获取 Teacher Top-p Tokens (Nucleus)

在 Student rollout 到达的状态 $s_t$ 下，分别获取完整词表概率分布 $p = \pi_T(\cdot \mid s_t)$ 与 $q = \pi_\theta(\cdot \mid s_t)$ 。  
按照 Teacher 概率从高到低排序，选取累计概率质量达到 $\rho$ 的最小集合（推荐设置 $\rho = 0.8$ ）：

$$
\mathcal{N}_t^\rho = \text{TopP}(p, \rho)
$$

> **目的**：首先聚焦 Teacher 最核心的 probability mass，避免无意义的长尾低概率 token 引入优化噪声。

---

### Step 2. 判定 Student 相对 Support 并构建 Deficit Set

对 Teacher nucleus 中的每个 token $i \in \mathcal{N}_t^\rho$ ，计算相对支持度比值：

$$
r_i = \frac{q_i}{p_i}
$$

设定相对 support 阈值 $\tau$ （如 $\tau = 0.1$ ）。若 $q_i / p_i \lt \tau$ ，则判定为欠覆盖 token（Deficit token），构建 Deficit Set：

$$
\mathcal{D}_t = \\{ i \in \mathcal{N}_t^\rho : \frac{q_i}{p_i} \lt \tau \\}
$$

---

### Step 3. 计算 Teacher-relative Coverage (路由指标)

定义 Student 对 Teacher 核心概率质量的覆盖度指标 $C_t$ ：

$$
C_t = \frac{\sum_{i \in \mathcal{N}_t^\rho} p_i \cdot \mathbb{I}[q_i / p_i \ge \tau]}{\sum_{i \in \mathcal{N}_t^\rho} p_i} = 1 - \frac{\sum_{i \in \mathcal{D}_t} p_i}{\sum_{i \in \mathcal{N}_t^\rho} p_i}
$$

- **物理意义**：表示 Teacher nucleus 中有多少比例的概率质量已获得 Student 的相对支持。 $C_t$ 作为动态调控两路损失的 Router。

---

### Step 4. 构造 Deficit-selective Residual-aware Forward KL Recovery Loss

为避免简单 Deficit-only FKL 导致整体质量分配畸变（推动欠覆盖集合总概率 $Q_D \to 1$ ），算法将 Deficit Set 以外的所有 token 合并为一个**残差类别（Residual Category）**：

$$
L_t^{REC} = \sum_{i \in \mathcal{D}_t} p_i \log \frac{p_i}{q_i} + (1 - m_D) \log \frac{1 - m_D}{1 - Q_D}
$$

公式具体由两项构成：

1. **Deficit-token FKL 项**（针对严重欠覆盖的 Teacher modes 进行直接的逐 Token 模式恢复）：

$$
L_{\text{deficit}} = \sum_{i \in \mathcal{D}_t} p_i \log \frac{p_i}{q_i}
$$

2. **Residual-category FKL 项**（将 Deficit Set 以外的所有 token 聚合成单一残差类别，显式约束剩余词表保留合理的总概率质量）：

$$
L_{\text{residual}} = (1 - m_D) \log \frac{1 - m_D}{1 - Q_D}
$$

其中质量统计项定义为：

$$
m_D = \sum_{i \in \mathcal{D}_t} p_i, \qquad Q_D = \sum_{i \in \mathcal{D}_t} q_i
$$

#### 梯度性质
对任意欠覆盖 token $i \in \mathcal{D}_t$ ，其对 logit 的梯度严格满足：

$$
\frac{\partial L_t^{REC}}{\partial z_i} = q_i - p_i, \quad \forall i \in \mathcal{D}_t
$$

- **优势**：
  1. 对欠覆盖 tokens 提供清晰无偏的 FKL 恢复梯度；
  2. 显式约束剩余词表保留合理的总概率质量 $(1 - m_D)$ ；
  3. 不强求剩余词表内部各 token 完全匹配 FKL，减少对已有良好模式的干扰。

---

### Step 5. 构造 Coverage-Adaptive 动态混合权重

设定覆盖度阈值 $\delta_C = 0.8$ ，最大恢复权重 $\alpha_{\max} = 0.8$ 。动态计算恢复权重 $\alpha_F(C_t)$ 与优化权重 $\alpha_R(C_t)$ ：

$$
\alpha_F(C_t) = \alpha_{\max} \cdot \text{clip}\left( \frac{\delta_C - C_t}{\delta_C}, 0, 1 \right)
$$

$$
\alpha_R(C_t) = 1 - \alpha_F(C_t)
$$

#### 策略调配表

| Coverage ($C_t$) | RKL 权重 $\alpha_R$ | Recovery FKL 权重 $\alpha_F$ | 调度策略 |
| :---: | :---: | :---: | :--- |
| $\ge 0.8$ | 100% | 0% | **Full RKL**：覆盖充足，全力进行 Mode-seeking 精细优化 |
| $0.6$ | 80% | 20% | **Hybrid**：覆盖良好，轻度补充恢复信号 |
| $0.4$ | 60% | 40% | **Hybrid**：平滑平衡模式恢复与精炼 |
| $0.2$ | 40% | 60% | **Recovery-heavy**：强化欠覆盖模式的梯度恢复 |
| $0.0$ | 20% | 80% | **Recovery-heavy**：最大程度恢复，底线保留 20% RKL 抑制 excess |

> **保留最低 20% RKL 的原因**：在模式恢复阶段仍需借助 RKL 抑制 Student 在无效低密度区域的概率溢出。

---

### Step 6. 最终 Token-level 联合损失

$$
L_t = \alpha_R(C_t) L_t^{RKL} + \alpha_F(C_t) L_t^{REC}
$$

其中：
- $L_t^{RKL} = \sum_{i=1}^{V} q_i \log \frac{q_i}{p_i}$ （始终在全词表上计算）；
- 路由判定、Deficit Mask 以及混合权重均应用 `stop-gradient`；
- 两路损失共享相同的 Student Softmax 概率分布。
