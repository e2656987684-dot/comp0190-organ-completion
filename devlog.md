# 开发日志 (devlog)

## 2026-06-25
- 目标：搭好项目骨架与 Git 仓库
- 做了：建仓库、目录结构、.gitignore、README、PROGRESS
- 结果：
- 决策：
- 下一步：
- 卡点：

## 2026-08-05
- 目标：数据从 50 对扩到 100 对之后，从"跑通"转向"提升效果"。
- 做了：
  - 数据准备：`notebooks/demo/MSN_train_skullfix.ipynb` 新增「0. 数据准备」cell，
    在 notebook 里直接调用 `src/data/prepare_skullfix.py`（不用再开终端切环境，
    `comp0190` 已并入 `comp0190-msn`），`N_SAMPLES` 默认改成 `0`（全部 100 对）。
    已跑过一次，`data/cache/skullfix_pairs_4096_6144.npz` 已是 100 对的缓存。
  - 复查了一次实际训练结果（`experiments/msn_skullfix/{run.json,history.csv}`），
    发现问题：`--minutes 3` 在 100 对数据下只够跑 19 个 epoch（比之前 40 对时測的
    "3分钟≈40epoch"少很多），且 val_loss 在 epoch 17 见顶（1.0950）之后 epoch 18
    已经回升（1.1946）才被 `--minutes` 掐停——没有任何机制检测到"已经过了最优点"，
    纯粹是运气好停在了没差太多的地方。`ModelCheckpoint(save_best_only=True)` 本身
    是对的，`best.h5` 存的确实是 epoch17 的权重，但训练进程自己不会提前收手。
  - 针对上面的问题，在 `src/models/train_skullfix.py` 里加了三处（对应基础提升
    清单里的 3/4/5 项），默认参数保持向后兼容，不加参数时行为和之前完全一样：
    - `--early-stop-patience`（默认 20，0 关闭）：加了
      `EarlyStopping(monitor="val_loss", restore_best_weights=True)`，
      val_loss 连续 N 轮不再变好就主动停，不用再赌 `--minutes` 蒙对时机。
    - `--n-folds` / `--fold`（默认 0，即维持原来的单次 `--val-frac` 随机切分）：
      100 个样本单次 80/20 切分的验证集只有 20 个颅骨，噪声偏大；加上这两个参数
      之后可以做 k-fold 交叉验证，多切几份取平均，而不是信一次运气。
    - `--run-name`（默认空，行为不变）：`--out-dir` 下的可选子目录，为了以后要
      重新扫收敛曲线（换几个 `--minutes` 值多跑几次）时，各次跑的
      `history.csv`/`best.h5` 不会互相覆盖。notebook 里读权重的 cell 硬编码的是
      `experiments/msn_skullfix/best.h5`，所以默认（不传 `--run-name`）时输出路径
      完全不变，不会破坏现有的评估 cell。
- 结果：（改动已过语法检查和 `--help` 检查，还没有实际重新跑训练验证效果，见下一步）
- 决策：这一轮只动了"基础提升"里风险最低、复用价值最高的三项（早停、k-fold、
  防覆盖的 run 目录），架构层面的改动和数据增强先不动，等这三项验证过再说。
- 下一步（按之前讨论的优先级，留作待办）：
  - 【基础，未做】给模型加正则化（dropout / AdamW 的 weight decay）——187M 参数
    对 80 个训练样本，过拟合目前几乎没有约束。
  - 【基础，未做】`cd_dcd_loss` 的 `dcd_weight` 目前硬编码 1.0
    （`msn_skullfix.py` `cd_dcd_loss`），可以暴露成 CLI 参数或做成随训练进度从
    0 斜坡升到 1。
  - 【基础，未做】用新加的 `--run-name` 实际跑一遍收敛曲线扫描（比如
    `--minutes` 取 5/10/20/30），更新 notebook 里那张已经过时的"3/5/7分钟"表。
  - 【基础，未做】数据增强：绕 SI 轴小角度旋转 + 轻微 jitter（注意大角度旋转会
    破坏颅骨在 LPS 坐标系下的方向先验）。
  - 【进阶，未做】用 `msn_downloads/MSN_weights3.h5`（作者发布的预训练权重）做
    微调而不是从零训练，100 个样本对 187M 参数依然是重度记忆化体量。
  - 【进阶，未做】解码器目前只吃编码器 max-pool 出来的一个全局向量
    （`build_decoder`），缺损颅骨里本来完好的表面也被整个重新生成，没有原始
    MSN 论文里"可见区域直接透传 + minimum density sampling 合并"那一步，这项
    如果做，预期收益在几个改动里最大，但工作量也最大。
  - 【进阶，未做】`PointSampler(mode="unique")` 是不重复随机采样，不是真正的
    FPS；数据准备阶段已经在用 `fpsample`，可以把同样的思路搬进 encoder 的
    `E-SG1`/`E-SG2` 下采样步骤。
  - 【进阶，未做】在 `paper()`（187M）和 `small()`（9.4M）之间找一个更匹配
    100 样本规模的中间容量，而不是默认焊死用最大配置。
  - 【数据工程，未做】接入 SkullBreak（README 里提到的目标数据集之一），从根本
    上缓解样本量不足，比任何单项建模改动的预期收益都大。
- 卡点：以上三项代码改动还没有实际跑一次训练验证是否达到预期效果（尤其是
  EarlyStopping 会不会因为 100 个验证样本本身噪声大而提前停得太早，
  patience=20 是拍的一个保守值，需要实跑后再调）。

- 追加：把 `--minutes`/`--epochs` 的默认值也重新配平了。之前 `--epochs` 默认
  10000、`--minutes` 默认 55，10000 从来没被跑到过、纯粹是摆设。现在
  `EarlyStopping` 已经是默认开启的真正停止信号，把两个默认值都调整为：
  `--epochs` 300（按 ~9.5s/epoch 算，全跑满约 47 分钟，仍然远高于预期的实际
  收敛点，只是不再是 10000 那种不现实的数字），`--minutes` 90（比 300 epoch
  需要的时间更宽松，正常情况下不会是它先触发，只在 EarlyStopping/epoch 上限
  都没起作用时兜底）。notebook 训练 cell 里的 `MINUTES` 变量也同步从 3 改成
  90，不然 cell 会一直显式传 `--minutes 3` 把脚本的新默认值覆盖掉。
  单折预计训练时长：EarlyStopping 触发前大概率在几十分钟以内结束（对着
  40 样本时"前20 epoch吃掉95%进步"的经验外推，100 样本单折不做 k-fold 时估计
  10~15 分钟量级，但这仍是未经验证的推测）；如果之后开 k-fold，总时长要乘以
  折数，且 `train_skullfix.py` 本身不会自动循环所有折，跑完整个 k-fold 需要
  手动改 `--fold`/`--run-name` 跑多次。

## 2026-08-05（补记：这次改动一度被"放弃更改"整体还原）
- 情况：`train_skullfix.py`、`devlog.md`（这份文件本身）、以及 notebook 里的
  「0. 数据准备」cell 和 `MINUTES` 改动，一度被整体还原回了改动前的版本（大概率
  是编辑器里误点了 Discard Changes，而不是 git 层面的操作——当时 `git status`
  显示这几个文件在 git 里从未被 commit 过，本来就该停留在"已修改未提交"状态）。
  发现后已经把上面记录的内容原样重新应用了一遍。
- 决策：这几个文件目前都还没有被 git 提交（`git status` 会显示为 modified）。
  如果不想再经历一次"改动消失"，跟用户确认后可以考虑先 `git add` + `git commit`
  一次，把这一轮改动落到 git 历史里，这样以后即使编辑器里误 discard，也能从
  git 历史里找回，而不是只存在于当前工作目录里。

## 2026-08-05（数据准备卡死：worker 内存无限增长 + OOM 重启死循环）
- 现象：`prepare_skullfix.py --workers 8` 跑了 18 分钟没结束，`.npz` 一直没出现
  （`np.savez` 是最后一步，所以中途看不到半成品）。用户感觉"卡住了"。
- 诊断（实测证据）：
  - worker 的 RSS 随它处理过的任务数**单调增长**，且新 worker 不断被补进来。
    抓到的一张快照：最初那批 worker（14:16 启动）已经涨到 7.2 GB / 7.5 GB，
    14:26/14:27/14:29/14:30 陆续新建的分别是 3.6 / 2.8 / 2.1 / 1.6 GB，
    而 14:34 刚生成的那个只有 3 MB —— 年纪越大占用越高，说明内存只涨不跌。
  - 本容器的 **cgroup 内存上限是 42.8 GiB**，不是 `free -h` 显示的 187 GiB
    （`free` 报的是宿主机，不是配额）。8 个这样的 worker 撑不到跑完就会撞上限，
    被 OOM 杀掉，`Pool` 再默默补一个新的 —— 于是陷入"杀掉→重启"的死循环。
  - 更致命的是 `pool.imap` 会**永久等待**随 worker 一起死掉的那个结果，
    所以这个任务不会自己结束，只会一直耗着。
- 根因：`Pool(args.workers)` 用了默认的 `maxtasksperchild=None`，worker 永不重启；
  每个任务瞬时要分配 248 MB 的 int32 体数据 + 114 万顶点 / 227 万面的网格 +
  采样中间量（峰值 1~2 GB），glibc 又不把释放的堆还给操作系统，于是 RSS 累积。
- 修复：`Pool(args.workers, maxtasksperchild=1)` —— 每做完一对就重启 worker，
  峰值被钉在"单任务工作集 × worker 数"（8 个约 16 GB，安全）。代价是每个任务多
  一次进程 spawn（约 0.2 秒），相对 2~3 秒的任务耗时可忽略。
- 同时修正了一个**之前给错的建议**：早先我按"48 逻辑核 / 内存充足"推荐把 workers
  开到 32~40，这是错的。这个负载是**内存带宽密集型**而非算力密集型，实测 24 对：
  独占 6.9s/对，4 workers 3.9s，8 workers 3.1s，12 workers 3.4s，24 workers 4.6s
  —— 12 比 8 慢、24 比 4 还慢，整台 24 核机器最多只能榨出约 2.3 倍加速。
  已把脚本 `--workers` 默认值和 notebook 的 `WORKERS` 都定在实测最优的 **8**，
  并把这组数字写进脚本 docstring，避免以后又凭核数拍脑袋调大。
- 另一个值得记的坑：tqdm 经 subprocess 管道打到 notebook 里时，`pool.imap` 按序
  回收结果 → 进度条会先静默十几秒再一次性跳几格，而它显示的**第一个速率**把全部
  启动开销都算到那 1 个样本头上（实测小样本跑出 `15.22s/it`，收敛后其实是
  `3.21s/it`）。之前误判"一对要一分钟"就是被这个读数骗了，不是真实吞吐。
- 明确不做：`marching_cubes(step_size=2)` + 体数据转 uint8 这组优化（实测能在多
  worker 场景快 1.6 倍，且精度损失已验证等同于采样噪声地板）**按用户要求不加**，
  保持现在的 step_size=1 / int32 路径不变。
- 卡点：卡死的那个进程组需要手动 kill（Claude 这边执行 kill 被权限策略拦下了）。

## 2026-08-05（预训练权重基线：新建 notebook + 一个静默失败的坑）
- 目标：把作者发布的 `MSN_weights3.h5` 在**已修复对齐**的数据上重跑一遍，
  作为和本项目从零训练结果对比的基线。
- 踩到并修正的一个错误判断（重要）：我先前建议"把预训练权重加载进
  `msn_skullfix.py` 的 `paper()`，两边走同一套代码"，**这是错的**。实测：
  `paper()` 与 `MSN_weights3.h5` 只有 3 个权重组能按名字匹配上
  （`D-OUT_lin`/`D1-IN`/`D2-IN`，共 32 组），而且
  `load_weights(by_name=True, skip_mismatch=True)` **不报错就返回** ——
  约 96% 的网络仍是随机初始化，会安静地产出垃圾预测。
  根因是层嵌套方式不同、不是形状不同：demo 的 `LBR` 把 Dense+ReLU 包在名为
  `E-IN_LBR1` 的嵌套 Model 里（存成 `E-IN_LBR1/E-IN_LBR1_lin/kernel`），
  重写版是顶层裸 Dense `E-IN_LBR1_lin`（找 `E-IN_LBR1_lin/kernel`），
  `by_name` 匹配不了。注意力块同理（`E-SA1` vs `E-SA1_Q/_K/_V`）。
  `msn_skullfix.py` docstring 里"paper() can load MSN_weights3.h5"那句话
  需要更正，目前先在 `msn_demo_arch.py` 的 docstring 里写清楚了。
- 做了：
  - 新增 `src/models/msn_demo_arch.py`：把 demo notebook 的架构定义
    （cells 4/6/8/10/12）**逐字**抽出来成模块。验证过它能以**严格模式**
    （不传 by_name / skip_mismatch）加载 `MSN_weights3.h5`，40/40 权重张量确实被改写。
    这个文件里的层名是有语义的，不要"顺手清理"。
  - 新增 `notebooks/MSN_baseline_pretrained.ipynb`：读已对齐的 `.npz`
    （不再碰 `.ply`，也**不再调 `normalize_point_cloud`**）、只评估训练时那
    20 颗验证集颅骨、指标乘各自的 `scale_mm` 出毫米、内置权重加载自检。
  - 更新 README 的仓库结构说明（原来那份是最初的骨架规划，和现状对不上），
    补了"两条数据路线（.npz vs .ply）分别是什么、该用哪条"的对照表。
- 结果（冒烟测试，3 颗验证颅骨）：预训练权重 CD_t 约 9.7 / 10.1 / 11.1 mm，
  本项目从零训练是 7.08 mm。完整 20 颗的数字要跑完 notebook 才有。
- 待确认：`MSN_weights3.h5` 训练时是否见过颅骨类数据。这决定了对比表里该写
  "zero-shot 迁移"还是别的表述，写进论文前必须查证原作者说明。
- 已知未解决：demo 的 `UniformSampler` 是有状态随机抽样，同一输入两次调用输出
  会有差异（训练 notebook 实测 1.03）。基线 notebook 里固定了全局种子，但要
  完全逐位可复现需要改成 stateless 抽样 —— 那属于改动 demo 架构，没做。

## 2026-08-05（流程决策：迭代期用单折，定稿后再跑 5 折）
- 决策：`--n-folds` 的代码已经就位，但**迭代期不开**，继续用默认的单次 80/20
  划分（seed 42）。等所有改动（正则化 / dcd_weight / 数据增强 / 可见区域透传 等）
  都定稿之后，再跑一次 5 折出最终数字。
- 理由：5 折是单折的 5 倍时间，每改一版就重跑 5 折纯属浪费。
- 副作用（必须记住，否则最后会误判）：
  - 单折验证集只有 20 颗，噪声不小 —— 上次曲线里 epoch 112→113 的 val_loss
    从 0.9243 跳到 0.9908（单轮 7%），折合 CD_t 约 0.4mm 的抖动。
    **迭代期只有 1mm 以上的变化才算可信，+0.3mm 级别的"改进"大概率是噪声。**
  - 反复对着同一批 20 颗决定"改动留不留"，本质是拿验证集做选择，改的次数多了
    会隐性过拟合到这 20 颗。**最终 5 折的数字大概率会比迭代期看到的差一些，
    这是正常的**，5 折那个才是真数字。
  - 迭代期保持 seed=42 不变，至少保证各版改动之间横向可比；不要中途换 seed。
