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
    "use_genque": true,
    "genque_choice": "2",
    "submit_command": "sbatch 2-vasp_gpu.sh",
    "done_file": "vasprun.xml"
  }
}
```

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

## 9. 测试

仓库包含三个小测试配置：

```bash
bash scripts/run_mace_al.sh run test_smoke/param.json test_smoke/machine.json
bash scripts/run_mace_al.sh run test_vasp/param.json test_vasp/machine.json
bash scripts/run_mace_al.sh run test_full/param.json test_full/machine.json
```

其中 `test_full` 会完成一个最小闭环：探索、VASP 标记、MACE 微调、导出最终模型。

