# MMSI 项目设置指南

## 项目信息
- **论文**: Modeling Multimodal Social Interactions: New Challenges and Baselines with Densely Aligned Representations (CVPR 2024 Oral)
- **作者**: Sangmin Lee, Bolin Lai, Fiona Ryan, Bikram Boote, James M. Rehg
- **论文链接**: https://arxiv.org/abs/2403.02090

## 项目日期
创建日期: 2026年3月9日

## 项目目录
```
/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project/
└── MMSI/
    ├── model.py           # 模型定义
    ├── train.py           # 训练脚本
    ├── test.py            # 测试脚本
    ├── dataloader.py      # 数据加载器
    ├── utils.py           # 工具函数
    ├── README.md          # 原项目说明
    ├── assets/            # 图片资源
    └── .git/              # Git 版本控制
```

## 环境要求
- Python 3.x
- PyTorch 2.0+
- transformers
- numpy

## 第一步：配置 Python 环境

### 方案 A: 使用 Conda（推荐）
```bash
# 创建新环境
conda create -n mmsi python=3.10
conda activate mmsi

# 安装 PyTorch（适配 GPU）
conda install pytorch::pytorch torchvision torchaudio -c pytorch -c nvidia

# 安装依赖
pip install transformers numpy
```

### 方案 B: 使用 Python venv
```bash
# 创建虚拟环境
python3 -m venv mmsi_env
source mmsi_env/bin/activate

# 安装 PyTorch（需要根据你的 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install transformers numpy
```

## 第二步：准备数据集

### 数据来源
项目需要以下数据集：

1. **基准数据集**（YouTube, Ego4D）
   - 下载链接: https://www.dropbox.com/scl/fo/fbv6njzu1ynbgv9wgtrwo/ANPk2TKqK2rl44MqKu05ogk
   - 需要下载后解压到项目目录中

2. **原始视频数据**
   - 访问: https://persuasion-deductiongame.socialai-data.org/
   - 用于获取原始视频内容（如果需要的话）

3. **关键点样本**（Aligned Player Keypoint）
   - 下载链接: https://www.dropbox.com/scl/fo/01rp8c126kc9014kbhvkg/AO2JvbsFuMd4WkkwzzOR06U
   - 包含预处理的关键点数据

4. **Werewolf 数据集**（HuggingFace）
   - 链接: https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/tree/main
   - 可选：用于社交推演游戏的特定数据集

### 数据目录结构
将下载的数据放在以下位置（根据 train.py 参数调整）：
```
/path/to/data/
├── txt_dir/              # 文本数据
├── txt_labeled_dir/      # 标注的文本数据
├── keypoint_dir/         # 关键点数据
├── meta_dir/             # 元数据
└── data_split_file       # 数据分割配置
```

## 第三步：运行训练

项目支持三个主要任务：

### 1. Speaking Target Identification (STI)
识别谁是说话对象
```bash
cd /mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project/MMSI

python train.py \
    --task 'STI' \
    --txt_dir '/path/to/txt_dir' \
    --txt_labeled_dir '/path/to/txt_labeled_dir' \
    --keypoint_dir '/path/to/keypoint_dir' \
    --meta_dir '/path/to/meta_dir' \
    --data_split_file '/path/to/data_split_file' \
    --checkpoint_save_dir './checkpoints' \
    --language_model 'bert' \
    --max_people_num 6 \
    --context_length 5 \
    --batch_size 16 \
    --learning_rate 0.000005 \
    --epochs 200 \
    --epochs_warmup 10
```

### 2. Pronoun Coreference Resolution (PCR)
解决代词指代问题
```bash
python train.py \
    --task 'PCR' \
    --txt_dir '/path/to/txt_dir' \
    --txt_labeled_dir '/path/to/txt_labeled_dir' \
    --keypoint_dir '/path/to/keypoint_dir' \
    --meta_dir '/path/to/meta_dir' \
    --data_split_file '/path/to/data_split_file' \
    --checkpoint_save_dir './checkpoints' \
    --language_model 'bert' \
    --max_people_num 6 \
    --context_length 5 \
    --batch_size 16 \
    --learning_rate 0.000005 \
    --epochs 200 \
    --epochs_warmup 10
```

### 3. Mentioned Player Prediction (MPP)
预测提及的参与者
```bash
python train.py \
    --task 'MPP' \
    --txt_dir '/path/to/txt_dir' \
    --txt_labeled_dir '/path/to/txt_labeled_dir' \
    --keypoint_dir '/path/to/keypoint_dir' \
    --meta_dir '/path/to/meta_dir' \
    --data_split_file '/path/to/data_split_file' \
    --checkpoint_save_dir './checkpoints' \
    --language_model 'bert' \
    --max_people_num 6 \
    --context_length 5 \
    --batch_size 16 \
    --learning_rate 0.000005 \
    --epochs 200 \
    --epochs_warmup 10
```

## 第四步：测试模型

使用 test.py 进行模型评估（需要已训练的模型检查点）：
```bash
python test.py \
    --task 'STI' \
    --checkpoint '/path/to/checkpoint.pt' \
    # ... 其他参数
```

## 注意事项

1. **数据路径**: 确保所有数据路径正确指向下载的数据位置
2. **GPU 内存**: 确保有足够的 GPU 内存（建议 16GB+）
3. **CUDA 版本**: 安装 PyTorch 时选择与服务器 CUDA 版本兼容的版本
4. **训练时间**: 200 个 epoch 的训练可能需要数小时到数天，建议使用 `nohup` 或 `screen` 后台运行
5. **检查点保存**: 模型权重会定期保存到 `--checkpoint_save_dir` 指定的目录

## 使用 GPU 服务器的建议

### 后台运行训练（推荐）
```bash
# 使用 nohup
nohup python train.py ... > training.log 2>&1 &

# 或使用 screen（更灵活）
screen -S mmsi_train
python train.py ...
# Ctrl+A 然后 D 来看离开会话

# 重新连接会话
screen -r mmsi_train
```

### 监控训练
```bash
# 实时查看日志
tail -f training.log

# 集群资源使用情况
gpu-status  # 或 nvidia-smi（如果可用）
```

## 项目链接总结
- GitHub 项目: https://github.com/sangmin-git/MMSI/tree/main
- HuggingFace 数据集: https://huggingface.co/datasets/bolinlai/Werewolf-Among-Us/tree/main
- 论文 arXiv: https://arxiv.org/abs/2403.02090

## 下一步
1. 下载所需的数据集
2. 根据你的 GPU 和数据位置调整训练命令中的参数
3. 开始训练或与我讨论任何问题
