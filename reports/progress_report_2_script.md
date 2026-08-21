# Progress report 2 — speaking script / 第二次进度汇报讲稿

Bilingual notes for `reports/progress_report_2.pptx` (7 slides), covering the
phase that starts at the 2026-08-07 ablation. One section per slide.
每页一节，中英对照。对应 `reports/progress_report_2.pptx`，7 页。

**Numbers / 数字出处.** Every figure quoted here comes from runs in
`experiments/`, evaluated through `src/eval/report.py` on the same 20 validation
skulls. All four ablation cells were trained under identical settings, so the
table is a like-for-like comparison.
本讲稿引用的每个数字都来自 `experiments/` 里的真实 run，用 `src/eval/report.py`
在同一批 20 颗验证颅骨上评估。消融的四格训练设置完全相同，是同口径比较。

**Pacing / 时间.** Roughly 10–12 minutes: about 1 minute on slide 1, 2–3 minutes
each on slides 2 and 3, 1–2 on slide 4, 2 on the outline, 1 on next steps.
约 10~12 分钟。第 1 页 1 分钟，第 2、3 页各 2~3 分钟，第 4 页 1~2 分钟，
大纲 2 分钟，最后一页 1 分钟。

---

## Slide 1 — Where things stand / 目前进展

**EN.** Quick reminder of the setup before the new material. The task is skull
completion — 4,096 points of a defective skull in, 6,144 points of a complete
skull out. SkullFix, 80 training and 20 validation skulls, and every run I show
today uses the same split, so nothing in the comparisons comes from a different
set of skulls.

One point on units. Each skull is normalised before it enters the network, but
every cloud carries its own scale factor, so I convert Chamfer and Hausdorff back
into millimetres before reporting. All numbers today are in millimetres.

Since the last report there are four things. I finished the loss ablation that
was still open. I added metrics that score the model only inside the defect
rather than over the whole skull. I measured which parts of the network are
actually being used. And I have started writing the thesis and started the
cross-validation runs.

**中文.** 先花一分钟回顾设定。任务是颅骨补全——输入 4096 个点的残缺颅骨，输出
6144 个点的完整颅骨。数据集 SkullFix，80 训练 / 20 验证，今天展示的每一次实验
都用同一个划分，所以对比里不会掺入"换了一批颅骨"这种因素。

单位说明一句。每颗颅骨进网络前都做了归一化，但每份点云都带着自己的尺度因子，
所以报告前我把 Chamfer 和 Hausdorff 都换算回毫米。今天所有数字都是毫米。

上次汇报之后有四件事：把还没做完的损失函数消融跑完了；加了只在缺损区内计算的
指标，而不是在整颗颅骨上算；测了网络里哪些部分是真正在起作用的；以及开始写论文、
开始跑交叉验证。

---

## Slide 2 — Loss ablation: is DCD still needed? / 损失消融：DCD 还需要吗？

**EN.** First, what the repulsion term is, since it drives this slide. It is a
hinge penalty on each point's nearest neighbours: if two points are closer than
2 millimetres, it pushes them apart, and if they are further apart it does
nothing. It is divided by that radius before squaring, so the loss is
dimensionless and the weight means the same thing no matter what scale the cloud
is in — that matters here because each skull has its own scale factor.

Last time repulsion had clearly fixed the clumping. What it left open was
whether DCD — the density-aware loss from the original paper — is still earning
its place. So I trained the two missing configurations, and now all four
combinations exist under identical settings.

Reading the table: each cell is Chamfer distance, then the percentage of
predicted points sitting closer than 2 millimetres to a neighbour. Chamfer plus
repulsion, bottom right, is the best cell, and it is also the simplest of the
four — no DCD term at all.

Three things to note. First, on whether these differences are real: repeating a
run with identical settings moves Chamfer distance by 0.004 millimetres, so the
0.07 millimetre spread across this table is well above noise. Second, DCD does
work on its own — it takes clumping from 13.6 down to 5.6 percent — but repulsion
alone reaches 1.3, and stacking the two gains nothing. So DCD can be dropped, and
that is a cleaner result than adding another term. Third, look along the Chamfer
column: accuracy barely moves across all four. Density and accuracy are being
controlled by different mechanisms, which is worth knowing.

**中文.** 先说 repulsion 是什么，这一页都建立在它上面。它是对每个点的最近邻做
hinge 惩罚：两个点距离小于 2 毫米就把它们推开，大于 2 毫米就什么都不做。平方前
除以那个半径，所以这个损失是无量纲的，权重在任何尺度下含义相同——这里很重要，
因为每颗颅骨都有自己的尺度因子。

上次 repulsion 已经明显解决了扎堆问题。当时留下的问题是：DCD——原论文那个密度感知
损失——还有没有存在的必要。所以我把缺的两个配置训练出来，现在四种组合在完全相同的
设置下都有了。

看表：每一格是 Chamfer 距离，然后是最近邻小于 2 毫米的预测点占比。右下角的
Chamfer + repulsion 是最好的一格，同时也是四个里最简单的——完全不含 DCD。

三点值得说。第一，这些差异是不是真的：同配置重复一次，Chamfer 距离只差 0.004 毫米，
所以表里 0.07 毫米的跨度远高于噪声。第二，DCD 单独确实有用——把扎堆率从 13.6% 降到
5.6%——但 repulsion 单独就能到 1.3%，两个叠加没有任何额外收益。所以 DCD 可以去掉，
而"证明可以去掉一项"比"再叠一项"是更干净的结果。第三，看 Chamfer 那一列：四个配置
的精度几乎不动。密度和精度是被不同机制控制的，这一点值得记下来。

---

## Slide 3 — Scoring inside the defect / 只在缺损区打分

**EN.** This slide is about a problem with how I was measuring, not with the
model. A score computed over the whole skull is dominated by the intact part —
and the intact part is exactly what the model can copy straight from its input.
So a whole-cloud number tells you mostly about the easy region, and the region
that matters clinically is a small fraction of it.

So I defined a defect region. A ground-truth point counts as being in the defect
when the nearest point of the *defective input* is more than 5 millimetres away.
I did not pick 5 arbitrarily — I measured the distribution of those distances and
it is clearly bimodal, with one peak at 2 to 3 millimetres, which is the shared
surface, a second peak past 15, which is the hole, and a valley between 5 and 6.
The threshold sits in that valley. It selects about 6 percent of the ground-truth
points, and that fraction is identical for every configuration, since it depends
only on the data.

The table shows two of the defect metrics for the same four configurations.
Coverage is how well the hole gets filled; precision is how accurate the points
that land there are.

Two conclusions. First, the improvement inside the hole is larger than over the
whole skull — about 17 percent against 12 percent — so the changes are helping
where the problem is genuinely hard, not just where it is easy. Second, and this
is the more useful one: coverage separates the configurations, but precision sits
between 2.89 and 2.96 for all four and tells you essentially nothing. So when I
report defect-region results, coverage is the metric to use, and I now have a
measured reason for that rather than a preference.

**中文.** 这一页讲的是我"怎么量"的问题，不是模型的问题。在整颗颅骨上算的分数会被
完好的部分主导——而完好的部分恰恰是模型可以直接从输入抄过来的。所以全点云的数字
主要反映的是容易的区域，而临床上真正要紧的那块只占其中一小部分。

于是我定义了缺损区。当一个真值点到**残缺输入**的最近点超过 5 毫米时，就算它在缺损区。
5 毫米不是随便定的——我量了这个距离的分布，它是明显的双峰：一个峰在 2~3 毫米，那是
共享表面；第二个峰在 15 毫米以上，那是洞；中间 5~6 毫米是谷底。阈值就取在谷底。
它选中约 6% 的真值点，而且这个比例在所有配置下完全相同，因为它只取决于数据。

表里是同样四个配置的两个缺损区指标。覆盖是洞被填得好不好，精度是落进去的点准不准。

两个结论。第一，洞里的改善幅度**大于**全点云——大约 17% 对 12%——说明这些改动是在
真正困难的地方起作用，不是只在容易的地方。第二，也是更有用的一条：覆盖能区分这四个
配置，而精度在 2.89 到 2.96 之间，基本什么都说明不了。所以我报告缺损区结果时以覆盖
为主指标，而现在这是有实测依据的，不是个人偏好。

---

## Slide 4 — A look inside the model / 看看模型内部

> ⚠️ 这一页讲**慢一点、短一点**。落点是"我知道了不该往哪儿花时间"，不是技术细节。
> 一分钟讲完就够，不要主动展开机制。

**EN.** One more thing I did this phase. The architecture is a transformer, so I
wanted to check whether the attention layers — the part it is named for — are
actually doing anything.

There are 16 attention blocks in the network. For 12 of them the answer is no:
they end up giving every point almost exactly the same weight. When the weights
are uniform, the block is not selecting anything — it is just averaging. Four of
them, in the second half of the decoder, do behave like real attention.

Why this is useful: it tells me where not to spend time. I had tried giving the
model more attention capacity, and the results got worse. Now I know why. So the
remaining effort goes into the loss functions and the evaluation, which is where
the measurable gains have actually come from.

**中文.** 这阶段还做了一件事。这个架构是 transformer，所以我想确认一下：它赖以命名的
注意力层，到底有没有在起作用。

网络里一共有 16 个注意力块。其中 12 个的答案是没有：它们最后给每个点的权重几乎完全
一样。权重均匀就意味着这个块没有在做选择，它只是在求平均。另外 4 个在解码器后半段，
表现是真正的注意力。

这件事的用处在于：它告诉我不该往哪儿花时间。我之前试过给模型加更多注意力容量，
结果反而变差。现在我知道原因了。所以剩下的精力放在损失函数和评测上——那才是实际
产生可测量收益的地方。

### 可能被问到的问题 / If asked

**Q：你怎么测出来"权重是均匀的"？**
把训练好的模型跑一遍，把注意力权重矩阵取出来看它的分布就行。均匀的话每个权重都等于
1 除以点数。
*EN: I export the attention weight matrices from the trained model and look at
their distribution. Uniform means every weight equals one over the number of points.*

**Q：那是不是说这个模型不适合这个任务？**
不能这么说。模型本身是有效的——指标一直在改善。只是有效的部分不是注意力，而是它的
特征提取和解码器。
*EN: Not quite. The model works — the metrics keep improving. It is just that the
part doing the work is the feature extraction and the decoder, not the attention.*

**Q：为什么会这样？**
最可能的原因是数据规模。这类方法通常在几万个形状上训练，我这里是几十个。
*EN: Most likely the data scale. These methods are usually trained on tens of
thousands of shapes; here it is tens.*

---

## Slides 5–6 — Thesis outline / 论文大纲

**EN.** The working title is "Point-Cloud Completion of Cranial Defects:
Density-Aware Losses and Defect-Region Evaluation" — which reflects where the
substantial results are: the loss work and the evaluation work.

Chapter 1 sets up the clinical problem, how cranial implants are designed today,
and why I work in point clouds rather than voxel grids — including what that
choice costs.

Chapter 2 is background: point-cloud completion from PCN onwards, the
transformer-based methods, and the multimodal model this work builds on, together
with the datasets in this area and their sizes.

Chapter 3 is method: the architecture as I have reimplemented it, the data
pipeline from meshes through to aligned point clouds, and how my training setup
differs from the published one.

Chapter 4 is the evaluation protocol — the metrics and why, the defect-region
metrics and how the region is defined, and what the point-cloud representation
itself limits. That last part matters: it is why my numbers cannot simply be
placed next to published voxel-domain results.

Chapter 5 is the loss functions: the density problem, the repulsion term, and the
ablation that settles which terms are needed.

Chapter 6 is a short chapter on which parts of the network are actually used, and
what that implies for where effort is worth spending.

Then discussion and conclusion.

**中文.** 暂定题目是 "Point-Cloud Completion of Cranial Defects: Density-Aware
Losses and Defect-Region Evaluation"——它反映的是实质结果所在：损失函数那部分和
评测那部分。

第 1 章交代临床问题、目前颅骨植入体是怎么设计的，以及为什么我用点云而不是体素网格，
包括这个选择的代价。

第 2 章是背景：从 PCN 开始的点云补全、基于 transformer 的方法、以及本工作所基于的
多模态模型，还有这个领域的数据集和它们的规模。

第 3 章是方法：我重新实现的架构、从网格到对齐点云的数据管线，以及我的训练设置与
已发表版本有哪些不同。

第 4 章是评测协议——用了哪些指标、为什么，缺损区指标和缺损区怎么定义，以及点云这种
表示方式本身的限制。最后这点很重要：它正是我的数值不能直接和已发表的体素域结果并排
的原因。

第 5 章是损失函数：密度问题、repulsion 项，以及那个确定哪些项需要、哪些可以去掉的
消融实验。

第 6 章是一个短章节，讲网络的哪些部分是真正被用上的，以及这对"精力该花在哪"意味着
什么。

然后是讨论和结论。

---

## Slide 7 — Next steps / 后续

**EN.** Three things.

Cross-validation is running. It is taking longer than a single run would, for two
reasons: the ablation left several configurations that all need covering, and I
have moved to a larger dataset. So the numbers are not ready yet, but the runs are
under way.

Writing has started, on the parts that are already settled and are not going to
change — the method chapter and the evaluation protocol.

And after that, a surface-reconstruction comparison, so that the point-cloud
numbers can be related to the ones clinical work is usually reported in. That is
the piece that would let me say something about how far this is from being
clinically usable.

**中文.** 三件事。

交叉验证在跑。比单次训练慢，有两个原因：消融留下了几个配置都要覆盖，以及我换成了
更大的数据集。所以结果还没出来，但实验已经在进行。

论文已经开始写了，写的是已经定下来、不会再变的部分——方法章节和评测协议。

之后是表面重建对照，这样点云的数值就能和临床工作通常报告的那套数值对应起来。
这一块做完才能说清楚"离临床可用还有多远"。
