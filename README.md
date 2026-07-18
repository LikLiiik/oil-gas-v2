# Dense Seismic-Well CLIP: 油气地球物理多模态特征融合

> 赛题 XH-202604："透视"地下油气藏——油气地球物理大模型的多模态特征融合
>
> 发榜单位：中国石油勘探开发研究院

## 1. 问题定义

地震和测井是油气勘探的两大核心技术手段，但两者存在本质差异：

| 维度 | 地震图像 | 测井曲线 |
|---|---|---|
| 模态 | 2D 图像（反射系数振幅） | 1D 时序（岩石物理属性） |
| 空间覆盖 | 横向连续，覆盖面广 | 单点测量，稀疏分布 |
| 物理含义 | 速度变化率（相对值） | 绝对岩石物理属性 |
| 分辨率 | 米级 | 厘米级 |

**核心挑战**：如何融合两种模态，使地震图像获得测井级别的岩石物理解释能力？

**赛题目标**：提出地震+测井统一表征技术，引入测井信息后地震图像的有利目标识别有显著提升。

## 2. 数据构建流程

基于 OpenSeisML 论文（Bhar et al., 2026）的 7 步数据治理管线：

```
┌──────────────┐    ┌──────────────┐    ┌───────────────┐
│ 1. 数据生成   │───→│ 2. 网格提取   │───→│ 3. RBF 速度建场 │
│ 合成3D速度模型 │    │ 包围盒+规则网格│    │ 多二次核RBF插值 │
│ + 井曲线+校深  │    └──────────────┘    └───────┬───────┘
└──────────────┘                                  │
       ↑                                          ↓
┌──────────────┐    ┌──────────────┐    ┌───────────────┐
│ 7. HDF5存储   │←───│ 6. FFT重采样  │←───│ 5. 准2D线提取  │←─┐
│ 256×512×12.5m│    │ 余弦渐变低通   │    │ 过井剖面       │  │
└──────────────┘    └──────────────┘    └───────────────┘  │
                                                            │
                                              ┌───────────────┐
                                              │ 4. 时深转换    │
                                              │ zₖ = Σ(vⱼ/2)Δt│
                                              └───────────────┘
```

**输出数据集规格**：

| 组件 | 内容 |
|---|---|
| 地震剖面 | 80 张，256×512 像素，12.5m 采样间距 |
| 测井曲线 | 80 口井 × 7 条曲线（GR, NPHI, RHOB, DT, RT, VEL, Depth） |
| 岩石物理标签 | 5 类：速度、孔隙度、岩性、密度、电阻率 |
| 格式 | HDF5（`openseisml_dataset_labeled.h5`，~60MB） |

## 3. 模型架构

### 3.1 整体设计

```
                   对比学习阶段（Stage 1）                任务监督阶段（Stage 2）
                   
   Seismic ──→ U-Net ──→ F_seis[H,W,C] ──→ Proj ──→ E_seis[H,W,D]
   (1,H,W)      │                                      │
                │                              密集 InfoNCE
                │                           对齐每个深度点
   Well    ──→ 1D Conv → F_well[L,C] ──→ Proj ──→ E_well[L,D]
   (6,L)                              ┌──────────────────┘
                                      │
                                      │ 梯度反向传播：
                                      │ 测井特征 ▸ 地震编码器
                                      │
              推理时 =================│========================
                                      │
   Seismic ──→ U-Net ──→ F_seis ──→ TaskHeads ──→ 逐像素岩石物理属性
   (无需测井)                                            │
                                          ┌───────────────┤
                                          │速度│孔隙度│密度│电阻率│岩性│
                                          └───────────────┘
```

### 3.2 编码器

**地震编码器 (SeismicUNet)** — 14.3M 参数

```
输入: (1, 256, 512) 地震剖面
  │
  ├── Stem: Conv7×7 stride2 → (32, 128, 256)
  ├── EncStage0: 2×ResBlock → (32, 64, 128)   ──── skip ──┐
  ├── EncStage1: 2×ResBlock → (64, 32, 64)    ──── skip ──┤
  ├── EncStage2: 2×ResBlock → (128, 16, 32)   ──── skip ──┤
  ├── EncStage3: 2×ResBlock → (256, 8, 16)    ──── skip ──┤
  ├── Bottleneck: 2×ResBlock → (512, 8, 16)               │
  ├── DecBlock3: deconv+skip3 → (256, 16, 32)  ←──────────┘
  ├── DecBlock2: deconv+skip2 → (128, 32, 64)  ←──────────┘
  ├── DecBlock1: deconv+skip1 → (64, 64, 128)   ←──────────┘
  ├── DecBlock0: deconv+skip0 → (32, 128, 256)  ←──────────┘
  └── Final: Conv1×1 → (128, 256, 512)
```

**测井编码器 (WellLogEncoder1D)** — 0.28M 参数

```
输入: (6, 256) 多通道测井曲线
  │
  ├── Stem: Conv1d stride2 → (64, 64)
  ├── 8×DilatedResBlock: dilation=[1,2,4,8,1,2,4,8]
  │   (保持深度分辨率，扩张卷积增大感受野)
  └── Conv1×1 → (128, 256)
```

### 3.3 密集对比损失

CLIP 式对称 InfoNCE，但应用到**每个深度点**而非整张图：

$$L_{\text{dense}} = -\frac{1}{B \cdot L} \sum_{b=1}^{B} \sum_{z=1}^{L} \log \frac{\exp(\text{sim}(e^s_{b,z}, e^w_{b,z}) / \tau)}{\sum_{z'=1}^{L} \exp(\text{sim}(e^s_{b,z}, e^w_{b,z'}) / \tau)}$$

- 正样本：同一剖面、同一深度点的地震-测井特征对
- 负样本：同剖面其他深度点、同 batch 其他剖面的所有深度点
- Batch=8, L=256 → 相似度矩阵 (2048, 2048) → 4.2M 负样本

### 3.4 任务预测头

```python
Shared: Conv3×3(128→128) + BN + ReLU
  ├── VelocityHead:  Conv... → Tanh×3 → 速度 (z-scored)
  ├── PorosityHead:  Conv... → Sigmoid×0.45 → 孔隙度
  ├── DensityHead:   Conv... → Tanh×3 → 密度 (z-scored)
  ├── ResistivityHead: Conv... → Softplus → 电阻率
  └── LithologyHead: Conv... → 3-class Softmax → 岩性
```

## 4. 训练策略

### 两阶段训练

```
Stage 1: 密集对比预训练 (60 epochs)
  ├── 无监督，无需标签
  ├── 优化 InfoNCE 损失
  ├── 学习率: 1e-4, warmup 5 ep, cosine decay
  └── 目标：对齐地震和测井特征空间

Stage 2: 联合任务监督 (40 epochs)
  ├── 加载 Stage 1 最佳权重
  ├── 对比损失 + 5 个岩石物理任务损失
  ├── 学习率: 5e-5
  └── 目标：从对齐的地震特征中预测岩石物理属性
```

### 训练细节

| 参数 | 值 |
|---|---|
| Batch size | 8（每 batch 2,048 个深度点对） |
| 优化器 | AdamW (β1=0.9, β2=0.999, wd=0.01) |
| 混合精度 | FP16 (GradScaler) |
| 梯度裁剪 | max_norm=1.0 |
| 温度参数 τ | 可学习，初始化 0.07 |
| 数据增强 | 水平翻转、高斯噪声、深度窗口裁剪 |
| 硬件 | 1× A100 80GB, ~2.5h 训练 |

## 5. 实验结果

### 5.1 跨模态对齐

| 阶段 | 损失 | 检索精度 |
|---|---|---|
| 初始 | 6.46 | 0.3%（≈随机） |
| Stage 1 终点 | 0.50 | 94.2% |
| Stage 2 终点 | 0.50 | 94.5% |
| 最终测试 | — | **98.7%** |

### 5.2 岩石物理属性预测

| 任务 | MAE | R² | 含义 |
|---|---|---|---|
| 速度预测 | 47.7 m/s | **0.989** | 1.8% 相对误差 |
| 密度预测 | 0.013 g/cm³ | **0.985** | 0.6% 相对误差 |
| 电阻率预测 | 1.48 ohm·m | **0.868** | 从地震"看"流体 |
| 岩性分类 | 93.9% Acc | — | 3 类（页岩/粉砂/砂岩） |
| 孔隙度预测 | 0.052 v/v | 0.302 | 合成数据中噪声大 |

### 5.3 关键发现

1. **测井信息不需要显式注入**。对比学习使地震编码器隐式内化了岩石物理约束，推理时仅需地震输入。
2. **密集对比学习解决了小样本问题**。80 个剖面 → 20,480 个深度点样本 → 每 batch 420 万个负样本。
3. **合成数据中地震和测井来自同一速度模型，不相关于简单线性关系**。地震测量速度变化率（反射系数 r≈0），测井记录绝对速度——模型学到了这个非线性映射。

## 6. 代码结构

```
├── openseisml_pipeline.py            # OpenSeisML 数据治理管线（7步）
├── visualize_dataset.py              # 数据集可视化（4张QC图）
│
├── dense_clip_seismic/               # 密集CLIP多模态融合框架
│   ├── config.py                     # 模型与训练配置
│   ├── models/
│   │   ├── seismic_unet.py           # U-Net 地震编码器
│   │   ├── well_encoder.py           # 1D 空洞卷积测井编码器
│   │   ├── dense_clip.py             # 密集CLIP模型 + InfoNCE损失
│   │   └── task_heads.py             # 岩石物理属性预测头
│   ├── data/
│   │   ├── dataset.py               # PyTorch Dataset + 数据增强
│   │   └── synthetic_labels.py      # 合成地质标签生成器
│   ├── generate_labeled_dataset.py   # 生成带标签HDF5数据集
│   ├── train_dense.py                # 两阶段训练脚本
│   └── evaluate_dense.py             # 评估与可视化
│
├── clip_seismic_well/                # 剖面级CLIP（早期版本）
├── openseisml_dataset.h5             # 40对基础数据集
├── openseisml_dataset_large.h5       # 200对扩展数据集
├── openseisml_dataset_labeled.h5     # 80对标注数据集（训练用）
├── dense_clip_checkpoints/           # 模型检查点
└── dense_clip_evaluation/            # 评估可视化
```

## 7. 使用方法

### 7.1 环境配置

```bash
conda create -n oil-gas python=3.10
conda activate oil-gas
pip install torch numpy scipy h5py matplotlib scikit-learn
```

### 7.2 生成训练数据

```bash
# 生成带岩石物理标签的数据集（80对）
python dense_clip_seismic/generate_labeled_dataset.py \
    --n-wells 80 \
    --output ./openseisml_dataset_labeled.h5 \
    --seed 42
```

### 7.3 训练模型

```bash
# 完整两阶段训练
python dense_clip_seismic/train_dense.py \
    --dataset ./openseisml_dataset_labeled.h5 \
    --epochs-s1 60 \
    --epochs-s2 40 \
    --batch-size 8 \
    --lr 1e-4 \
    --save-dir ./dense_clip_checkpoints
```

### 7.4 评估推理

```bash
python dense_clip_seismic/evaluate_dense.py \
    --checkpoint ./dense_clip_checkpoints/best_s2_joint.pt \
    --dataset ./openseisml_dataset_labeled.h5 \
    --output-dir ./dense_clip_evaluation
```

### 7.5 单张剖面推理

```python
import torch
from dense_clip_seismic.models.dense_clip import DenseSeismicWellCLIP
from dense_clip_seismic.config import DenseCLIPConfig

model = DenseSeismicWellCLIP(DenseCLIPConfig())
ckpt = torch.load("./dense_clip_checkpoints/best_s2_joint.pt", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 仅需地震剖面，无需测井
seismic = torch.randn(1, 1, 256, 512)  # 替换为实际数据
features, _ = model.encode_seismic(seismic)
predictions = model.task_heads(features)

# predictions["velocity"]  → (1, 1, 256, 512)  速度图
# predictions["porosity"]  → (1, 1, 256, 512)  孔隙度图
# predictions["density"]   → (1, 1, 256, 512)  密度图
# predictions["resistivity"] → (1, 1, 256, 512) 电阻率图
# predictions["lithology"] → (1, 3, 256, 512)  岩性分类
```

## 8. 后续工作

1. **接入真实 UK NDR / 赛方 10TB+ 数据**：用真实 SEG-Y + LAS + checkshot 替换合成数据生成器
2. **扩展到 3D**：用 3D U-Net + 3D 卷积测井编码器处理完整三维地震体
3. **半监督传播**：利用对比学习后的特征相似度，从有井位置向无井位置传播岩石物理属性
4. **引入地质约束**：在损失函数中加入地层连续性、断层位移等物理约束
5. **不确定性量化**：用扩散模型或贝叶斯方法对岩石物理预测提供置信区间

## 参考文献

- Bhar, I. et al. (2026). *OpenSeisML: Open Large-Scale Real Seismic and well-log Dataset for Generative AI*. arXiv:2605.20539v1.
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. PMLR.
- Erdinc, H.T. et al. (2024). *Generative Geostatistical Modeling from Incomplete Well and Imaged Seismic Observations with Diffusion Models*. arXiv:2406.05136.
