# RCPR：基于鲁棒共识原型细化的自监督共显著目标检测
​
> **基于鲁棒共识原型细化的自监督共显著目标检测设计与实现**
​
本项目面向**共显著目标检测（Co-Salient Object Detection, CoSOD）**任务，在自监督设置下（不使用像素级人工掩码）实现了 **RCPR（Robust Consensus Prototype Refinement，鲁棒共识原型细化）** 框架。项目以自监督 ViT（DINO ViT-B/8）的 patch-token 特征为基础，在 **推理阶段** 对图像组的共识原型进行闭环净化，从而在 CoCA、Cosal2015、CoSOD3k 三个公开基准上稳定提升检测质量。
​
项目同时提供一个轻量网页 demo：上传至少 2 张含共同目标的图片，即可在线查看原图、Mask 与叠加可视化结果。
​
---
​
## 一、研究背景与动机
​
共显著目标检测需要从**一组相关图像**中找出既在单图中显著、又在组内具有共同语义的目标。现有高性能方法大多依赖像素级掩码监督，标注成本高、泛化受限。
​
本项目采用**自监督**路线：利用预训练 ViT 的 patch-token 特征和跨图像对应关系生成初始共显著响应，无需密集标注。但自监督设置存在一个核心问题——**初始伪掩码噪声大**：背景纹理、边缘、局部高对比区域和非共同前景容易被错误激活。若直接在含噪伪掩码内平均特征，得到的共识原型会被噪声牵引（本项目称之为**共识原型坍缩**），导致后续区域筛选继续放大错误。
​
RCPR 的目标即是：**在不增加训练标注和可训练参数的前提下，利用 token 级特征一致性修正初始伪掩码噪声，得到更可靠的共识原型。**
​
---
​
## 二、方法概述（RCPR）

​
| 模块 | 名称 | 作用 |
| --- | --- | --- |
| **ACRE** | Adaptive Consensus Reweighted Estimator | 以 token 与共识原型的余弦一致性为依据迭代重加权，抑制噪声 token，缓解均值原型被污染的问题（核心模块） |
| **RTG-TopK** | Risk-controlled Top-K | 在「置信 / 质量保留 / 原型漂移」三道门控保护下做权重稀疏化，避免硬截断误删真实目标 token |
| **Soft-RPF** | Soft Region Prototype Feedback | 将第一轮原型评分以软硬混合方式反馈给初始权重，净化后进入第二轮 ACRE，兼顾边界召回与噪声抑制 |
| **Tau2** | 边界通道阈值 | 第二阶段使用受控放宽阈值 τ₂ = clip(τ₁ − δτ)，挽回被残余噪声压低的真实共同目标区域 |
​
**整体流程**：冻结 DINO ViT-B/8 提取 patch-token 特征 → 跨图像对应 + 置信门限得到初始响应 w₀ → 第一轮 ACRE（内嵌 RTG-TopK）得到原型 P₁* → Soft-RPF 反馈净化 → 第二轮 ACRE 得到 P₂* → Tau2 阈值输出预测掩码。
​
> 特点：RCPR **不引入新的可训练参数、不新增解码器**，全部细化都在推理阶段的 token 嵌入空间完成。
​
---
​
## 三、主要实验结果
​
主干特征与评估协议在所有数据集上保持一致，使用固定超参数（在 CoCA 上网格搜索后固定，其余数据集不再调参）。下表中 SCoSPARC 为本组在同一管线下复现的 baseline，RCPR 为加入原型净化模块后的结果（U = 无监督/自监督）。
​
| 方法 | 监督 | CoCA MAE↓ | CoCA F_max↑ | CoCA E_max↑ | CoCA S_α↑ | Cosal2015 F_max↑ | CoSOD3k F_max↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCoSPARC（复现 baseline） | U | 0.090 | 0.608 | 0.787 | 0.713 | 0.866 | 0.827 |
| **RCPR（本组）** | U | **0.087** | **0.622** | **0.794** | **0.720** | **0.868** | **0.831** |
​
- 在干扰物较多的 **CoCA** 上提升最明显：F_max 0.608 → 0.622，MAE 0.090 → 0.087。
- 相比无监督方法 US-CoSOD，CoCA 上 F_max 由 0.546 提升到 0.622（相对约 +13.9%）。
- 消融实验表明 **ACRE 是主要性能来源**，RTG-TopK 提供安全的稀疏化增益，Soft-RPF 改善目标召回，Tau2 进一步降低复杂图像组的 MAE。
​
> 评价指标：MAE、最大 F-measure（F_max）、最大 E-measure（E_max）、结构度量 S_α。
​
---
​
## 四、项目结构
​
```text
SCoSPARC-main课设版/
├─ models.py / rcpr.py / dataset.py / vision_transformer.py   # 模型与 RCPR 核心实现
├─ train.py / test.py                                         # 训练与推理/评估入口
├─ eval_metrics.py / loss.py / util.py / utils.py             # 评价指标、损失与工具
├─ cosod_backend.py                                           # 网页 demo 后端（/api/cosod 接口）
├─ 前端/code.html                                             # TailwindCSS 前端页面
├─ start_cosod_web.bat / stop_cosod_web.bat                   # Windows 一键启停脚本
├─ vis_compare_5col.py / vis_compare_coca.py                  # 定性可视化对比脚本
├─ requirements.txt / self_requirements.txt                   # 依赖
├─ checkpoints/  models/  datasets/  predictions/             # 权重 / 数据 / 输出（部分不入库，见下）
└─ CV_Course_Design.pdf                                       # 课程设计报告
```
​
---
​
## 五、环境与依赖
​
- Python + PyTorch（自监督 ViT 特征与 CoSOD 评估流程基于 PyTorch 生态）。
- 参考依赖：`requirements.txt`、`self_requirements.txt`。
- 本项目使用的 Conda 环境路径示例：
​
```powershell
D:\Anaconda\Anaconda\envs\scosparc
```
​
---
​
## 六、数据集与权重
​
### 数据集
​
- 评估基准：**CoCA**（80 组 / 1295 张）、**Cosal2015**（196 组 / 2015 张）、**CoSOD3k**（160 组 / 3316 张）。
- 训练数据：**COCO9213 + DUTS-Class** 组合（共 17463 张、65 类）。
- `datasets/` 与 `predictions/` **不上传到 GitHub**：
  - 网页 demo 通过上传图片临时推理，不依赖仓库内预置数据集；
  - 运行原始测试脚本前，请在本地放置 `datasets/CoCA`、`datasets/Cosal2015`、`datasets/CoSOD3k`；
  - `predictions/` 为推理输出目录，运行后本地自动生成。
​
### 权重（仓库只保留推理实际使用的两个）
​
- `models/dino_vitbase8_pretrain.pth`（DINO ViT-B/8 预训练主干，较大，建议用 Git LFS 管理）
- `checkpoints/baseline运行出的checkpoints/model_combo_base8-136_0.7291838924090067.pt`
​
其余 `*.pt` / `*.pth` 权重默认通过 `.gitignore` 排除。
​
---
​
## 七、运行方式
​
### 1）训练 / 复现评估
​
```powershell
# 训练（ViT 主干冻结，仅优化跨图像注意力相关部分）
python train.py
​
# 在 CoCA / Cosal2015 / CoSOD3k 上推理与评估
python test.py
```
​
### 2）定性可视化对比
​
```powershell
python vis_compare_5col.py
python vis_compare_coca.py
```
​
### 3）网页 Demo（可选）
​
Windows 下双击根目录的 `start_cosod_web.bat`，脚本会启动后端并打开：
​
```text
http://127.0.0.1:8765/code.html
```
​
也可手动启动：
​
```powershell
cd C:\COSOD\SCoSPARC-main课设版 && D:\Anaconda\Anaconda\envs\scosparc\python.exe cosod_backend.py --host 127.0.0.1 --port 8765
```
​
使用流程：打开网页 → 上传至少 2 张含共同目标的图片 → 点击“开始体验” → 查看原图、Mask 与叠加结果。关闭服务双击 `stop_cosod_web.bat`。
​
---
​
## 八、团队成员与分工
​
| 成员 | 学号 / 班级 | 主要分工 |
| --- | --- | --- |
| 许以诺 | 2023112349 / 智能 2023-02 班 | 自监督 CoSOD 文献调研、RCPR 方法设计、核心模块分析、实验结果整理与报告撰写 |
| 陈康 | 2023112364 / 智能 2023-02 班 | 算法代码实现、实验复现与评估、图表整理、LaTeX 排版与报告修改 |
​
---
​
## 九、局限与后续方向
​
- 依赖自监督 ViT 特征质量；若主干无法区分共同目标与背景，原型细化只能缓解噪声。
- 主要在 token 级 / 区域级细化，边界精度仍可能不如端到端监督分割模型。
- 部分超参数靠网格搜索确定，未来可研究自适应策略。
- 后续计划：尝试 DINOv2 / MAE 等主干、结合边界细化模块、完成 **MindSpore** 迁移版本。
​
---
- US-CoSOD：Unsupervised and semi-supervised CoSOD via segmentation frequency statistics, WACV 2024.
- DINO：Emerging properties in self-supervised vision transformers, ICCV 2021.
- Deep ViT Features as dense visual descriptors, 2021.
​
> 完整参考文献与方法细节见仓库内课程设计报告 `CV_Course_Design.pdf`。
