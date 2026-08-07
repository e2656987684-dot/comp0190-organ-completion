# Progress report — speaking script / 进度汇报讲稿

Bilingual notes for `progress_report.pptx`. One section per slide.
每页一节，中英对照。

Figures and numbers all come from `src/eval/make_report_figures.py`, which reads
the real runs — regenerate it and the deck stays in sync.

---

## Slide 1 — Task and evaluation baseline / 任务与评测基线

**EN.**
The task is skull completion: the input is a defective skull as 4,096 points, the
output a complete one as 6,144 points. On the left is the ground truth with the
missing region marked in red; on the right, the points our model places inside
that same region. Grey is surface that was already in the input.

Before improving anything I needed something to compare against, so I ran the
authors' released weights on my own data — same 20 validation skulls, same metric
definitions, same aligned point clouds. They score 10.71 mm Chamfer distance and
1.428 DCD. My model, trained from scratch on skulls, reaches 6.41 mm and 0.676:
40% and 51% better respectively.

One detail worth flagging. The source paper reports DCD = 1.41269 on its own
validation set. Their weights measure 1.42772 on my data — a 1% match. That
confirms my metric implementation is a faithful port and the weights are loading
correctly. It also says something about the comparison itself: their model has
not failed on skulls, it performs on skulls almost exactly as it does on its own
data. My gain comes from specialisation, not from fixing a defect in their model.

**中文。**
任务是颅骨补全:输入是 4096 个点的残缺颅骨,输出 6144 个点的完整颅骨。左图是真值,
红色标出缺失区域;右图是我们模型在同一区域生成的点。灰色是输入里本来就有的表面。

在做任何改进之前,我需要一个可比的基准,所以我把作者发布的权重放到自己的数据上跑了一遍
——同一批 20 颗验证颅骨、同一套指标定义、同一份已对齐的点云。它得到 10.71mm 的
Chamfer 距离和 1.428 的 DCD。我从零训练的模型是 6.41mm 和 0.676,分别好 40% 和 51%。

有个细节值得说。原论文在它自己的验证集上报的 DCD 是 1.41269,而它的权重在我的数据上
实测是 1.42772,相差 1%。这验证了我的指标实现是忠实移植、权重加载也正确。它同时说明了
这个对比的性质:**他们的模型在颅骨上并没有失效**,表现和在自己数据上几乎一样。
我的提升来自专精化,不是修复了它的缺陷。

---

## Slide 2 — The real gap is point density / 真正的差距在点密度

**EN.**
My original plan, and my supervisor's suggestion, was to add a smoothness penalty
— the predicted surface looked bumpy. Before implementing it I measured how bumpy
it actually was, and the answer changed the plan.

Local roughness comes out at 0.736 for ground truth against 0.760 for the
prediction. Essentially equal. There is nothing there to win.

What is genuinely different is how the points are spaced. Ground truth is
farthest-point sampled, so no two points are closer than 3 mm and the figure on
the left is a uniform green. The baseline model, on the right, has 12.8% of its
points sitting within 2 mm of a neighbour — those are the dark spots. Those
points are wasted: they cover no surface that a neighbour does not already cover.
The spacing coefficient of variation is 0.389 against ground truth's 0.145.

So the direction changed from smoothness to density, on the basis of measurement
rather than what the renders looked like.

**中文。**
我原本的计划,也是导师的建议,是加一个平滑惩罚项——预测的表面看起来坑坑洼洼。
但在动手之前我先量了一下到底有多不平,结果改变了计划。

局部粗糙度真值是 0.736,预测是 0.760,基本持平。这里没有可捡的便宜。

真正有差别的是点的疏密。真值是最远点采样出来的,任意两点间距都不小于 3mm,
所以左图是均匀的绿色。右边的基线模型有 **12.8% 的点距离邻点不到 2mm**,也就是那些暗斑。
这些点是浪费的:它们覆盖的表面邻点已经覆盖了。间距变异系数是 0.389,而真值是 0.145。

所以方向从"平滑"改成了"密度",依据是测量结果,而不是渲染图看上去的样子。

---

## Slide 3 — Why the density-aware loss cannot fix it / 为什么密度感知损失治不了它

**EN.**
The obvious response is to lean harder on DCD, the density-aware Chamfer distance
the original project trains with. I tried that, and it barely moved. This slide
is why.

DCD multiplies a distance factor by a density factor, one over count to the
lambda. Count is how many ground-truth points match the same predicted point, and
it comes from an argmin — a piecewise-constant function. Its gradient with
respect to point position is exactly zero. I verified this: TensorFlow returns
None for that path.

So DCD can re-weight the Chamfer gradients, but it can never apply a force that
pushes two predicted points apart. The diagram shows the case it is blind to: two
predictions sitting on top of each other, each matched by a different ground-truth
point. Count is one for both, so DCD charges nothing at all — and this is the
dominant form of the clumping I measured.

Two consequences. Tuning DCD's hyper-parameters was never going to solve density,
which explains why those experiments failed. And a repulsion term is not
redundant with DCD — it supplies exactly the gradient DCD structurally lacks.

**中文。**
最直接的反应是加大 DCD 的力度——原项目就是用这个密度感知 Chamfer 距离训练的。我试了,
几乎没动。这一页解释为什么。

DCD 是距离因子乘以密度因子,也就是 1 除以 count 的 λ 次方。count 是有多少个真值点匹配到
同一个预测点,而它来自 argmin——一个分段常数函数。**它对点位置的梯度恰好为零**。
我验证过:TensorFlow 在这条路径上返回 None。

所以 DCD 只能给 Chamfer 的梯度重新加权,**它永远无法产生把两个预测点推开的力**。
图里画的就是它看不见的情况:两个预测点重合在一起,各自被不同的真值点匹配,
两边的 count 都是 1,DCD 一分钱都不罚——而这正是我测到的扎堆的主要形态。

两个后果。第一,调 DCD 的超参本来就不可能解决密度问题,这解释了那几次实验为什么失败。
第二,repulsion 项和 DCD **不重复**——它提供的正是 DCD 结构上缺失的那个梯度。

---

## Slide 4 — Two changes that worked / 两个真正有效的改动

**EN.**
Two things moved the numbers, and I ran a controlled experiment to separate them.

The first was a configuration bug. The learning-rate scheduler and early stopping
both watch validation loss, but the scheduler's patience was 40 while early
stopping's was 20 — so training always ended first and the learning rate had
never once been reduced, in any experiment. Every run before this was trained at
a flat rate. Fixing it took Chamfer distance from 7.22 to 6.40 mm and let
training run to 279 epochs instead of 133. The dotted lines in the middle panel
are the drops.

The second was the repulsion loss — a hinge that penalises predicted points
closer than 2 mm to each other, with a gradient that vanishes once they are far
enough apart, so it stops fighting Chamfer instead of pushing indefinitely.

To attribute these properly I ran a third configuration with the learning-rate
fix but no repulsion. The result: all of the Chamfer improvement comes from the
learning rate — repulsion contributes nothing there, 6.40 against 6.41. What
repulsion does is density: clumping falls from 5.6% to 1.4%, against ground
truth's 0.0%, at no accuracy cost. So it is a density component, not an accuracy
component, and I will describe it that way.

The honest headline is that the largest single gain in this phase came from
correcting one number in a callback, not from loss design.

**中文。**
有两件事真正推动了指标,我专门做了对照实验把它们拆开。

第一件是一个配置 bug。学习率衰减和早停盯的是同一个信号——验证损失,但衰减的 patience 是
40、早停是 20,所以训练永远先结束,**学习率在任何一次实验里都从未被下调过**。
此前每一轮都是恒定学习率跑完的。修好之后 Chamfer 距离从 7.22mm 降到 6.40mm,
训练轮数从 133 涨到 279。中间那张图上的虚线就是每次下调的位置。

第二件是 repulsion 损失——一个铰链项,惩罚彼此距离小于 2mm 的预测点,
一旦分开够了梯度严格归零,所以它不会像其它形式那样无休止地推、和 Chamfer 拔河。

为了正确归因,我跑了第三个配置:修学习率但不加 repulsion。结果是:
**Chamfer 的提升全部来自学习率**,repulsion 在这一项上没有贡献,6.40 对 6.41。
repulsion 起作用的是密度:扎堆率从 5.6% 降到 1.4%,真值是 0.0%,而且不付出精度代价。
所以它是一个**密度组件而不是精度组件**,我会这样表述它。

诚实的标题是:本阶段最大的单项收益来自改正一个回调里的数字,而不是损失函数设计。

---

## Slide 5 — Results / 结果

**EN.**
The full table, on 20 validation skulls, with millimetres converted using each
skull's own scale.

Chamfer distance 7.22 to 6.41 mm. HD95, the 95th-percentile worst-case error,
7.24 to 6.21 — I added this because Chamfer is a mean and averages away a single
bad region, which for implant design is exactly what matters. Clumping 12.85% to
1.36%, essentially reaching ground truth. Note that dcd_l2, the DCD
hyper-parameter experiment, barely moves anything — consistent with the mechanism
on slide 3.

I also added F-score at two thresholds, because that is what the source paper
reports and it lets me put my numbers directly beside theirs. Their full model,
trained on 200,000 clouds across 240 classes, gets 0.937 at the loose threshold.
Their small-data run, 4,800 shapes, gets 0.892. Mine, on 80 skulls of a single
class, is 0.928 — between the two, with three orders of magnitude less training
data. At the strict threshold I am behind both, which is where the remaining work
is.

One caveat I want to state rather than hide: DCD and F-score are comparable across
projects because the definitions and normalisation match, but Chamfer distance is
not. Under this codebase's definition the paper's reported CD would fall 35 times
below the point spacing, which is physically impossible — they are almost
certainly using a squared-distance variant. So I do not put those side by side.

**中文。**
完整的表格,20 颗验证颅骨,毫米数用每颗颅骨自己的尺度换算。

Chamfer 距离从 7.22 降到 6.41mm。HD95,也就是 95 分位的最坏情况误差,从 7.24 降到 6.21
——我加这个指标是因为 Chamfer 是均值,会把某一处的严重偏差平均掉,而对植入体设计来说
那恰恰是最要命的。扎堆率从 12.85% 降到 1.36%,基本达到真值水平。注意 dcd_l2 那一行,
也就是 DCD 超参实验,几乎什么都没动——这和第 3 页的机制是一致的。

我还补了两个阈值下的 F-score,因为原论文报的就是这两个,补上之后我的数字才能和它们
并排放。他们的完整模型在 240 个类别、20 万个点云上训练,宽阈值下是 0.937;
他们的小数据量版本,4800 个形状,是 0.892。我的是 0.928——**介于两者之间,
而训练数据少了三个数量级**。严格阈值下我落后于两者,那正是后面工作的空间。

有一点我想说明而不是藏起来:DCD 和 F-score 可以跨项目比,因为定义和归一化都对得上;
但 Chamfer 距离**不能**。按本代码库的定义,论文报的 CD 会比点间距还小 35 倍,
物理上不可能——他们几乎肯定用的是平方距离的变体。所以我没有把这两个并排。

---

## Slide 6 — Next steps / 后续工作

**EN.**
Three things are planned.

First, ablate DCD away entirely. Density is now handled by repulsion, the
hyper-parameter experiments produced nothing, and slide 3 explains why. If
removing it changes nothing, three losses collapse to two — a cleaner result than
stacking terms.

Second, k-fold cross-validation. Every number I have shown is a single 20-skull
split. The code is in place and has never been run. This has to happen before
anything is written up.

Third, defect-region-restricted metrics. Most of the skull surface is already
present in the input, so the current metrics are diluted by how well the model
reproduces what it was handed. Restricting them to the defect measures the thing
that actually matters. This is an evaluation change only — no retraining.

Three more are under consideration rather than committed.

Focusing the model on the defect: the dataset ships implant labels that I have
never used. Training on them directly would put all of the model's capacity on
the hard part and match the AutoImplant evaluation protocol — but it is a large
change and the metrics would no longer be comparable with everything so far.

Point-to-plane attraction: an alternative route to my supervisor's smoothness
goal. Chamfer pulls each prediction toward a discrete sampled point, which lets
points scatter either side of the true surface; pulling toward a fitted local
plane instead removes that. It is orthogonal to repulsion and I can verify it with
tooling I already have.

And a Poisson reconstruction control. My current visualisation is deliberately
robust to uneven density, which means it cannot show why density matters
clinically. Poisson reconstruction is not — running it would connect the density
metric to actual usability, and I expect to be asked about that.

**中文。**
计划中的有三件。

第一,把 DCD 整个消融掉。密度现在由 repulsion 接管,超参实验什么也没跑出来,
第 3 页解释了原因。如果去掉它结果不变,三个损失就能收缩成两个——这比堆叠损失项
是更干净的结论。

第二,k 折交叉验证。我展示的每一个数字都来自单次 20 颗颅骨的划分。代码早就写好了,
从来没跑过。这件事必须在动笔之前完成。

第三,缺损区限定的指标。颅骨表面大部分在输入里本来就有,所以现在的指标被
"模型复现已知部分的能力"稀释了。把指标限定到缺损区,量的才是真正要考核的东西。
这纯粹是评测侧的改动,不用重新训练。

另外三件是设想,还没有确定要做。

**让模型聚焦缺损区**:数据集里带了 implant 标注,我一直没用过。直接以它为训练目标,
可以把模型全部容量放在难的部分,而且和 AutoImplant 的评测口径一致——
但改动量很大,而且指标会和此前所有结果不可比。

**点到面引力**:通往导师那个平滑目标的另一条路。Chamfer 是把预测点拉向某个离散的采样点,
这允许点散布在真实表面两侧;改成拉向拟合出来的局部平面就能消除这一点。
它和 repulsion 正交,而且我现有的工具就能验收。

**Poisson 重建对照**:我现在的可视化方法对密度不均是刻意鲁棒的,这意味着它无法展示
密度为什么在临床上重要。Poisson 重建不是——跑一次就能把密度指标和实际可用性连起来,
我预计会被问到这个问题。
