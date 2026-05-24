# MACE-AL 中文手册

## 1. 简介

MACE-AL 是一个面向 MACE foundation model 微调的主动学习自动工作流。
整体逻辑参考 DP-GEN：

```text
初始 committee
-> 模型驱动探索
-> 不确定性筛选
-> DFT/VASP 标记
-> 数据集合并
-> MACE committee 微调
-> 继续迭代
-> 导出最终生产模型
```

生产入口是：

```bash
maceal run param.json machine.json
```

在当前集群上推荐用包装脚本：

```bash
bash scripts/run_mace_al.sh run param.json machine.json
```

## 2. 文件结构

```text
param.json       科学参数和工作流参数
machine.json     机器、队列、module、VASP、POTCAR 参数
mace_al/         Python 包源码
scripts/         集群入口脚本
templates/vasp/  VASP 输入模板
docs/            中英文手册
```

运行输出在：

```text
cache/Generation-N/
final_models/
```

## 3. `param.json`

`param.json` 控制科学问题本身。

主要部分：

- `project`：seed 结构、训练集、测试集、MACE 初始大模型。
- `mace`：MACE 微调参数。
- `explore`：ASE-MD 探索参数和不确定性阈值。
- `select`：每轮最多选择多少结构进入 DFT。
- `dft`：VASP 模板目录和 POTCAR 元素映射。
- `run`：迭代代数和自动化策略。

示例：

```json
{
  "project": {
    "seed_file": "./data/seeds.extxyz",
    "train_file": "./data/train.extxyz",
    "foundation_models": [
      "/home/qhwu/mace_mod/mace-mpa-0-medium.model",
      "/home/qhwu/mace_mod/mace-omat-0-medium.model"
    ]
  },
  "run": {
    "max_generations": 3,
    "submit_dft": true,
    "wait_dft": true,
    "train_next_generation": true
  }
}
```

## 4. `machine.json`

`machine.json` 只放机器相关参数，便于同一套 `param.json` 换机器运行。

示例：

```json
{
  "mace": {
    "env_script": "module load anaconda && conda activate mace",
    "device": "cuda"
  },
  "explore": {
    "device": "cuda"
  },
  "dft": {
    "potcar_root": "/home/qhwu/PBE64",
    "use_genque": false,
    "submit_script": "./templates/vasp/submit_vasp.sh",
    "submit_command": "sbatch {submit_script}",
    "done_file": "vasprun.xml"
  }
}
```

推荐的队列模式是由用户自己提供 Slurm/PBS 提交脚本，并写到
`dft.submit_script`。MACE-AL 会把这个脚本复制到每个候选结构目录，并用
`{submit_script}` 填充 `dft.submit_command`。`genque` 只作为可选兼容路径，
只有显式设置 `dft.use_genque: true` 时才会调用。

如果做很小的本地测试，也可以直接在 bash 里跑 VASP：

```json
{
  "dft": {
    "direct_command": "module purge && module load vasp && vasp_std > vasp.out 2>&1"
  },
  "run": {
    "run_dft_direct": true,
    "submit_dft": false
  }
}
```

## 5. 数据格式

训练集使用 ASE `extxyz`，需要：

- `atoms.info["REF_energy"]`
- `atoms.arrays["REF_forces"]`
- 可选 `atoms.info["REF_stress"]`

seed 结构不需要标签。

## 6. 运行方式

生成演示数据：

```bash
bash scripts/run_mace_al.sh init-demo
```

启动自动主动学习：

```bash
bash scripts/run_mace_al.sh run param.json machine.json
```

查看状态：

```bash
bash scripts/run_mace_al.sh run-report param.json machine.json -v
```

重新导出最终模型：

```bash
bash scripts/run_mace_al.sh export param.json machine.json
```

## 7. 断点续跑

每个阶段完成后会写 `.done` 标记：

```text
cache/Generation-N/.explore.done
cache/Generation-N/.select.done
cache/Generation-N/.prepare_dft.done
cache/Generation-N/.collect_dft.done
```

如果 VASP 提交后程序停止，等作业完成后重新运行同一个命令：

```bash
bash scripts/run_mace_al.sh run param.json machine.json
```

程序会从第一个未完成阶段继续。

## 8. 最终模型

完整迭代完成后，生产模型导出到：

```text
final_models/
```

`manifest.json` 会记录最终 generation 和导出的模型文件。

默认情况下，重新导出会清理 `final_models/` 里不属于当前 manifest 的旧 `.model`
文件，避免生产环境误选旧代模型。这个行为不删除 `cache/Generation-N/`，旧代训练、
探索和 DFT 目录应当作为训练痕迹保留。

如果只想跑到某一代训练结束并导出模型，不想继续做该代的探索/DFT，可以设置：

```json
{
  "run": {
    "max_generations": 21,
    "stop_after_generation_train": 20
  }
}
```

这会在 Gen-20 committee 训练完成后停止并导出模型。

## 9. 自动绘图

MACE-AL 会自动生成诊断图：

- `cache/plots/generation_XXXX_exploration_uncertainty.png`：每代探索阶段的不确定性随 MD step 和相空间分布。
- `cache/plots/generation_XXXX_selection_map.png`：每代集中保存的 select 图。
- `cache/Generation-N/select/selection_map.png`：committee 不确定性分布图，被选中的结构会高亮。
- `cache/Generation-N/mace/committee/training_metrics.png`：MACE 训练日志中的能量/力验证误差。
- `cache/plots/generation_summary.png`：每轮候选结构数、选中结构数、DFT 标记结构数。

可以随时重新生成图：

```bash
bash scripts/run_mace_al.sh run-report param.json machine.json -v
```

## 10. 训练痕迹

每次收集 VASP 标签后，MACE-AL 会从累计训练集 `data/train.extxyz` 自动重建轻量
训练痕迹：

```text
cache/Generation-N/recovered_labels.extxyz
cache/recovered_generation_summary.tsv
cache/recovered_generation_summary.json
```

这些文件记录每一代实际进入训练集的 DFT 标记结构。它们不能替代原始
`OUTCAR`、`vasprun.xml`、探索轨迹和训练 checkpoint；生产项目仍应保留完整
`cache/Generation-N/` 目录。

如需手动重建：

```bash
bash scripts/run_mace_al.sh rebuild-traces param.json machine.json
```

## 11. 测试和代表性示例

仓库保留三个示例配置：

```bash
bash scripts/run_mace_al.sh run test_smoke/param.json test_smoke/machine.json
```

`test_smoke` 是快速冒烟测试，默认停在 VASP 输入准备后。

`test_representative_full` 是已经用于验证工作流的完整 BaFeF4 本地闭环：

```bash
bash scripts/run_mace_al.sh --root test_representative_full init-representative-demo --train-count 20 --test-count 4 --seed-count 2
bash scripts/run_mace_al.sh run test_representative_full/param.json test_representative_full/machine.json
bash scripts/run_mace_al.sh run-report test_representative_full/param.json test_representative_full/machine.json -v
```

它会完整执行 MACE/CUDA 探索、直接运行 VASP 标记、收集标签、训练下一代 MACE、
生成图像并导出最终模型。

MACE-AL 会在收集标签前检查 VASP 电子自洽是否收敛。出现 `EDIFF was not reached`
或 `electronic self-consistency was not achieved` 的作业会写入 `failed_jobs.txt`，
不会进入训练集。

`test_hfo2_production` 是更重的生产级 HfO2 验证示例：

```bash
bash scripts/run_mace_al.sh run test_hfo2_production/param.json test_hfo2_production/machine.json
bash scripts/run_mace_al.sh run-report test_hfo2_production/param.json test_hfo2_production/machine.json -v
```

这个示例从 12 原子 Pca21 HfO2 seed 和 MPA/OMAT 两个 MACE foundation model
出发，探索 16 个候选结构，选择 8 个结构进入 VASP，只有检测到 `ediff is reached`
标记后才收集标签，并导出两个 Generation-1 模型。VASP 模板使用更严格的设置：
`ENCUT=600`、`EDIFF=1E-6`、`NELM=240`、`ALGO=All`、`LASPH=.TRUE.`、
`LREAL=.FALSE.` 和 `2x2x2` Gamma-centered k 点。
