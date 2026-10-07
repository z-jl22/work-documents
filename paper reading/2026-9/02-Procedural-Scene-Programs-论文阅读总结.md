# Procedural Scene Programs for Open-Universe Scene Generation: LLM-Free Error Correction via Program Search

## 1. 背景：声明式declarative与命令式imperative布局的矛盾

近期方法通常采用 **declarative paradigm**：LLM 写出“物体 A 在物体 B 左边”“椅子不能碰撞桌子”等关系，再交给全局 constraint solver 求解。其优点是约束表达清晰，但场景规模变大时，约束数量通常随物体数量近似二次增长，求解时间和失败概率都会增加。

本文采用 **imperative paradigm**：LLM 按程序顺序放置物体，后续物体的位置是此前物体、场景或共享参数的函数：

$$
\text{object}_i.\text{attribute}
= f(\text{previous objects},\text{scene},\text{shared variables}).
$$

例如，不直接给每把椅子写绝对坐标，而是用循环和间距变量生成一排椅子。这样更符合 LLM 的逐步生成方式，也能用更紧凑的表示表达复杂场景。

## 2. PSDL：Procedural Scene Description Language

### 2.1 程序语义

PSDL 是嵌入 Python 的 DSL。程序接收 scene template，包括场景尺寸、对象列表和对象属性；程序通过 side effects 更新对象的 position、size 和 facing，最终得到布局。

PSDL 的核心表达包括：

```python
chair.max.x = table.min.x - 0.1
chair.center.y = table.center.y
chair.min.z = scene.min.z
chair.facing = table
```

这里的重点是右侧表达式不是固定坐标，而是具有语义的几何关系：椅子位于桌子左侧、与桌子 y 方向对齐、落在地面上并朝向桌子。**PSDL的核心就是使用程序化的语言去定义物体之间的相对位置关系**。

### 2.2 Expression sharing

PSDL 允许通过变量共享表达式：

```python
d = 2.0
for i, c in enumerate(cols):
    c.center.x = scene.center.x + i * d
```

变量 `d` 同时控制所有列的间距。如果纠错搜索把 `d` 从 2.0 调为 1.8，所有列会一致移动。论文认为这会缩小 search space，并使程序搜索更可能保持原始场景的整体结构。

### 2.3 Control flow

循环和条件语句可以表达 rows、symmetric placement、repeated objects 等结构。相较于逐个写坐标，循环中的 stride 具有隐式共享效果，也更容易通过少量参数进行修复。

## 3. 错误分类与纠错目标

论文区分两种错误：

### 3.1 Exceptions

包括 hallucinated function、错误参数、index out of range 等，使程序在运行时直接失败。本文对这类错误的处理仍主要是丢弃当前程序、再次请求 LLM 生成。

### 3.2 Layout errors

程序可以运行，但布局违反物理或几何要求，例如：

- objects overlap；
- objects go out of scene bounds；
- objects float above the floor；
- object support constraints are violated；
- facing direction 不满足语义关系；
- 后放置的物体堵住门或破坏原先已建立的关系。

本文主要研究第二类错误，因为它们可以通过明确的几何损失函数验证，而不需要再次调用 LLM。论文将一个布局记为 $L$，并定义综合损失 `loss(L)`。这不是单一的“碰撞计数”，而是由四类可量化的几何/物理约束组成。

### 3.3 四类布局损失

#### 1. Out-of-Bounds Loss

对每个物体，计算其 bounding cuboid 超出 scene boundary 的最大线性距离。如果物体完全位于场景边界内，该项为 0。该损失解决桌子、墙、物品等越出场景 cuboid 的问题。

#### 2. Overlap Loss

对每一对物体，计算其 bounding cuboids 交集体积的立方根：

$$
L_{\mathrm{overlap}}(o_i,o_j)
=\sqrt[3]{\mathrm{Vol}(\mathrm{AABB}_i\cap\mathrm{AABB}_j)}.
$$

使用立方根是为了把体积量纲转换为长度量纲，使其与其他距离型损失更容易共同优化。对于门和窗，论文使用 expanded collision boxes，将门扇打开空间等功能性区域也纳入碰撞检测，从而避免家具虽然没有碰到门本体，却阻挡门开启。

#### 3. Standing Loss

对于标记为 `STANDING` 的物体，计算物体 cuboid 的底面与最近可支撑它的水平表面之间的距离。物体悬空时该项为正；物体正确站在地面、桌面或其他水平支撑面上时，该距离应接近 0。

#### 4. Mounted Loss

对于标记为 `MOUNTED` 的物体，计算其可安装面（通常是背面或侧面）与最近可支撑它的垂直表面之间的距离。例如挂在墙上的画、壁挂物体，如果没有贴近墙面，就会产生 mounted loss。

因此，布局损失可以概念性地写作：

$$
\mathrm{loss}(L)
=L_{\mathrm{out\mbox{-}of\mbox{-}bounds}}(L)
+L_{\mathrm{overlap}}(L)
+L_{\mathrm{standing}}(L)
+L_{\mathrm{mounted}}(L).
$$

### 3.4 为什么不能只最小化 loss

如果只最小化 `loss(L)`，搜索器可能采用破坏语义的“投机修复”。例如，为了消除餐桌和椅子的重叠，可以把整张桌子移动到房间另一个空旷区域；几何上没有碰撞，但原先“椅子围绕餐桌”的布局意图被破坏。

因此作者把目标定义为：在消除布局错误的同时，尽量让修改后的程序和布局接近 LLM 的原始结果：

$$
P^*=\arg\min_P\left[\mathrm{loss}(L)+d(P,P_0)\right],
$$

其中 $P_0$ 是 LLM 生成的原始程序，$P$ 是修改后的程序，$L_0$ 和 $L$ 分别是二者执行得到的布局。距离项进一步分解为：

$$
d(P,P_0)=d_{\mathrm{edit}}(P,P_0)+d_{\mathrm{OT}}(L,L_0).
$$

#### Program edit distance

$d_{\mathrm{edit}}(P,P_0)$ 是把原始程序变成新程序所需的最短 elementary edit 序列长度。实验中 elementary edits 主要包括：

- rewrite constant expressions；
- rewrite direction expressions。

这是一个有意的限制：作者观察到 LLM 最常见的布局错误是数值参数不合适或 facing direction 设置错误，因此优先在这两个低维空间中搜索，而不是任意重写程序结构。

#### Layout optimal transport distance

程序编辑距离仍不能完全反映物体移动了多少，因此论文加入 layout-level optimal transport distance：

$$
d_{\mathrm{OT}}(L,L_0)
=\min_{f\in\mathcal{F}}
\sum_{o\in O(L)}
\mathrm{Vol}(o)\,
\left\|\mathrm{center}(o)-\mathrm{center}(f(o))\right\|_2.
$$

这里 $O(L)$ 是布局中的物体集合，$\mathcal{F}$ 是保持 object category 的双射集合，$f(o)$ 是原始布局中与 $o$ 对应的同类别物体。体积 `Vol(o)` 作为质量权重，意味着大物体的移动代价更高，小物体可以在必要时进行更大调整。该项鼓励系统保留大型场景元素的位置，同时允许椅子、装饰物等小物体局部移动。

综合来看，`loss(L)` 负责“有没有几何/支撑错误”，$d_{\mathrm{edit}}$ 负责“程序改了多少”，$d_{\mathrm{OT}}$ 负责“场景中的物体实际移动了多少”。三者共同实现“修复错误但不改变原始布局意图”。

## 4. 程序优化流程

优化流程如下：

1. 执行 LLM 生成的 PSDL program，得到原始程序 $P_0$ 和布局 $L_0$。
2. 计算 `Out-of-Bounds`、`Overlap`、`Standing` 和 `Mounted` 四类损失。
3. 找到程序中的 numerical constants 和 facing/direction expressions。
4. 为每个常量随机生成局部候选：将常量乘以 $\pm 4Y$，其中 $Y\sim U[-1,1]$；每个常量最多采样 10 个 edits。
5. 对 direction expression 枚举四个 cardinal directions：`X_NEG`、`X_POS`、`Y_NEG`、`Y_POS`，每个方向表达式最多形成 4 个候选。
6. 执行每个候选程序，并计算局部目标：

   $$
   f(L)=\mathrm{loss}(L)+d_{\mathrm{OT}}(L,L_0).
   $$

7. 选择使 $f(L)$ 降低最多的候选，形成局部搜索序列 $P_0,P_1,P_2,\ldots$：

   $$
   P_{i+1}=\arg\min_{P\in\mathcal{N}(P_i)} f(L).
   $$

8. 当可用 edit 无法使目标下降超过一个小阈值时终止。论文平均每个场景只需约 **7.13 次调整**。

与直接调整每个物体的坐标相比，PSDL 搜索是在 **procedural parametrization space** 中进行，变量和循环保证了很多候选程序天然保留原有结构。论文在最终优化中使用的是 `loss + d_OT` 的局部目标，而完整的程序相似性目标还包含 $d_{\mathrm{edit}}$；后者通过限制 elementary edits 的类型和数量来实现。

## 5. 评价指标

论文不直接把传统几何指标当作完整质量判断，因为无碰撞不等于符合 prompt。例如，一个布局可能没有重叠，但物体种类、相对位置或整体场景语义不符合文本描述。因此作者将评价设计为 **pairwise preference evaluation**，而不是为单个场景预测一个绝对分数。

### 5.1 LLMCompare 指标

给定一个 scene prompt 和两个布局的 rendered images，评价器要求多模态 LLM：

1. 分别列出两个布局相对于 prompt 的 **pros and cons**；
2. 综合 scene plausibility、object arrangement 和 prompt appropriateness；
3. 在最后判断 layout A 或 layout B 哪一个更好。

如果自动评价器的选择与人工多数票一致，则记为一次 agreement。论文将该指标称为 **LLMCompare**，并设置了一个不要求先输出 pros/cons 的消融版本 `LLMCompare (no +/-)`。

### 5.2 与其他自动指标的比较

论文还比较了两个用于 text-to-image 评价的指标：

- **VQAScore**：询问 VQA 模型“图像是否描绘了该 prompt”，使用输出 token `yes` 的概率作为分数；
- **Davidsonian Scene Graphs（DSG）**：先从 prompt 生成一组依赖关系图和 yes/no 问题，再由 VQA 系统回答并聚合 yes 的比例。

结果显示，VQAScore 和 DSG 与人工多数判断的一致率都接近随机水平，原因是只要图像中包含正确的物体类别，VQA 模型即使面对糟糕布局也可能判断场景类型正确；DSG 生成的 yes/no 问题也不一定能区分两个布局的空间质量。相较之下，要求 LLM 先列出 pros/cons 能迫使评价器关注布局关系，而不只是识别“有没有这个场景”。

Table 3 的人工一致率为：

| Automated metric | 与人工多数判断的一致率 |
|---|---:|
| **LLMCompare** | **77.1%** |
| LLMCompare（no pros/cons） | 70.0% |
| VQAScore | 58.6% |
| DSG | 50.7% |

因此，LLMCompare 相比不带 pros/cons 的版本提高 7.1 个百分点，说明“先解释、后选择”是评价设计中的有效组成部分。

## 6. 实验设置与结果

### 6.1 Benchmark 与 Prompt 构成

论文构造了一个开放词汇场景 prompt benchmark：

- **Ours prompt set**：70 个由作者整理的 prompts；
- **Holodeck prompt set**：52 个来自 MIT Scenes 数据集的 scene types，另加入 Holodeck qualitative examples 中的 14 个较长 prompts；
- 两组 prompt 合计 84 个不同场景描述，但部分分析按照论文定义的 70 个 Ours prompts、Holodeck prompts 和组合后的 66 个 Holodeck-related prompts 分开展示；
- prompt 覆盖 indoor/outdoor、realistic/fantastical、structured/chaotic 等类型；
- `Complex` 类别定义为长度至少 4 个 words 的 prompt，用于检验更详细文本描述下的表现。

生成对象来自场景模板和 object retrieval 模块。所有方法在尽可能一致的对象集合和渲染设置下比较，重点比较 layout quality，而不是不同物体检索器造成的差异。

### 6.2 对比方法

实验比较以下四种 LLM-based scene layout generation methods：

1. **Ours**：LLM 生成 PSDL procedural program，再通过 iterative LLM-free error correction 进行 program search。
2. **DeclBase**：Aguina-Kang et al. 提出的 declarative layout generation approach。
3. **Holodeck**：Holodeck 的 `Constraint-based Layout Design Module`。
4. **FlairGPT**：Littlefair et al. 提出的 declarative layout generation approach。

论文没有把 LayoutGPT 作为主要 baseline，因为作者认为它在比较对象中被 Holodeck 和 DeclBase 严格支配，加入该对比不会提供新的信息。

### 6.3 评价维度

实验分别回答四个问题：

- **Human preference**：人类是否更喜欢本文方法生成的布局？
- **Automatic preference**：LLMCompare 能否复现人工偏好？
- **Scaling with object count**：对象数量增加后方法是否仍然有效？
- **Error correction ablation**：PSDL、局部搜索、gradient descent 和 LLM self-repair 的错误率与耗时有何差异？

### 6.4 人工偏好实验

论文进行 **two-alternative forced-choice perceptual study**。共招募 10 名大学生参与者，分成两组，每组对应一个比较实验。每位参与者观看 70 个 comparisons；每个 comparison 包含一个 scene prompt、两个随机顺序呈现的 layout images，以及要求选择哪个布局更好的问题。判断标准是整体 scene plausibility 和与 prompt 的 appropriateness。

对每个 comparison，论文采用参与者的 majority vote 作为人工“gold standard”。结果为：

- **Ours w/ EC vs. DeclBase**：本文方法被偏好 **82.9%**；
- **Ours w/ EC vs. Holodeck**：本文方法被偏好 **94.3%**；
- **Ours w/ EC vs. Ours w/o EC**：带 error correction 的版本被偏好 **74.3%**；
- **Ours w/o EC vs. DeclBase**：没有纠错的 PSDL 版本仍被偏好 **61.8%**。

这组结果区分了两个贡献：PSDL procedural representation 本身已经优于 DeclBase，而加入 error correction 后又进一步提升了布局质量。

### 6.5 自动指标结果

Table 4 报告自动评价器偏好本文方法的比例。这里的 `Ours` 使用 70 个 Ours prompts；`Holodeck` 包含 MIT Scenes 和 Holodeck qualitative prompts。结果为：

| Comparison | All | Ours prompts | Holodeck prompts | Simple | Complex |
|---|---:|---:|---:|---:|---:|
| Ours vs. DeclBase | 76.5% | 77.1% | 75.8% | **77.8%** | 68.4% |
| Ours vs. Holodeck | 82.4% | **90.0%** | 74.2% | 82.9% | 78.9% |
| Ours vs. FlairGPT | **88.2%** | 85.7% | **90.9%** | 87.2% | **94.7%** |

从结果看，本文方法在所有 prompt 来源和复杂度上都保持优势；对 DeclBase 的优势在 complex prompts 上有所下降，但仍达到 68.4%。与 Holodeck 比较时，在作者自建 prompts 上达到 90.0%，说明其方法对开放式场景描述尤其有效。

Table 5 进一步按场景物体数量统计本文方法相对 Holodeck 的偏好率：

| 每个场景的物体数 | 场景数量 | Ours preference |
|---:|---:|---:|
| 10–20 | 9 | 66.7% |
| 20–30 | 45 | 80.0% |
| 30–40 | 37 | 72.9% |
| 40+ | 42 | **95.2%** |

40 个以上物体时达到 95.2%，说明 PSDL 的变量共享和循环结构在大规模、重复性强的场景中更能降低布局生成和纠错难度。

### 6.6 Error correction 消融实验

Table 6 比较不同 error correction module 的 preference rate、每个场景平均剩余错误数和平均运行时间：

| Error correction method | Preference rate | 平均剩余错误数 | 平均时间 |
|---|---:|---:|---:|
| No solver | 60.0% | 12.3 | 0 s |
| Gradient descent | 61.4% | **0.5** | 2.6 s |
| LLM self-repair | 68.1% | 5.7 | 106.7 s |
| Local search, basic imperative | 71.4% | 0.8 | 25.2 s |
| Local search, PSDL（ours） | — | 1.1 | **9.3 s** |

PSDL local search 平均剩余 1.1 个错误，相当于纠正约 91% 的错误；gradient descent 的剩余错误数更低，但它可以独立移动每个物体，容易破坏原有结构，因此 preference rate 不一定更高。PSDL 的关键优势是以更低代价在 procedural parameter space 中修复，同时保留变量共享和布局关系。

### 6.7 运行时间

本文方法的平均耗时由三部分构成：

| Pipeline stage | 平均时间 |
|---|---:|
| Scene template generation | 9.5 s |
| PSDL layout program generation | 19.2 s |
| Program-search error correction | 9.3 s |
| **Total** | **约 38 s** |

DeclBase 场景平均约 40.8 s，包括 10 s declarative program generation、21.3 s solver，以及共享的 template generation。因此本文方法虽然增加了纠错步骤，但整体速度与声明式系统相当；PSDL search 也比 basic imperative program search 的 25.2 s 快约 3 倍，原因是 loops 和 shared variables 降低了搜索空间维度。

## 7. 局限性

- local search 目前主要改数字常量和四种 cardinal facing，不能处理 object identity swap 或 x/y 轴整体理解错误。
- $L(P)$ 主要表达碰撞、越界、支撑和朝向，难以表达 line-of-sight、walkability、affordance 和整体审美。
- VLM evaluator 可能存在自身偏差。
- fixed object retrieval module 和离散朝向限制了开放世界程度。
- 如果程序中最初的高层语义就错了，局部数值搜索很难修复，需要更高层的语义重写或结构搜索。
