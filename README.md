# Qwen-1.8B 微调项目（全量微调 + LoRA）

本项目提供基于 [Qwen/Qwen-1.8B](https://huggingface.co/Qwen/Qwen-1.8B) 的监督微调（SFT）训练框架，支持 LoRA 微调 和 全量微调 两种方式，适配 HuggingFace Transformers，适合自定义中文对话任务。

---

## 📁 项目结构

```
qwen_finetune_demo/
├── configs/                  # 训练配置文件（支持 LoRA 控制）
│   └── train_config.yaml
├── data/                     # 微调数据（prompt + response 格式）
│   ├── train.jsonl
├── model/
│   ├── dataset.py            # 自定义数据加载器
│   └── trainer.py            # LoRA / 全量微调逻辑
├── scripts/
│   └── run_train.sh          # 一键训练脚本（bash）
├── prepare_data.py          # 自动下载并转换数据集为 jsonl
├── main.py                  # 训练主入口
├── requirements.txt         # 所需依赖包
└── README.md
```

---

## 🔧 安装依赖

建议使用 Python 3.10 + 虚拟环境 + CUDA 11.7/12 环境：

```bash
pip install -r requirements.txt
```

---

## 📥 下载并准备数据（HuggingFace 格式）

```bash
python prepare_data.py
```

数据将保存为：`data/train.jsonl`，每条样本为：
```json
{"prompt": "请分析一下这家公司的财报。", "response": "根据数据，营收同比增长10%。"}
```

---

## ⚙️ 配置训练参数（configs/train_config.yaml）

```yaml
model_name_or_path: Qwen/Qwen-1.8B
lora: true            # true: 使用LoRA微调；false: 全量微调
output_dir: ./output_qwen
train_batch_size: 1
learning_rate: 2e-5
...
```

---

## 🚀 开始训练

```bash
python main.py
```

或者使用脚本：
```bash
bash scripts/run_train.sh
```

训练中间结果和最终模型将保存在 `output_qwen/` 中。

---

## 🧠 支持功能
- ✅ LoRA 微调（内置 peft 支持）
- ✅ 全量微调
- ✅ 可替换 HuggingFace 自定义数据集
- ✅ 配置灵活（YAML 配置）

---

## 🔮 推理（示例）

后续可添加 `inference.py`，用于加载模型和生成回答。

---

## 📌 数据格式要求（兼容 Alpaca 格式）

```json
{"instruction": "请回答以下财务问题。", "input": "某公司净利润增长为负，原因可能是什么？", "output": "可能是营业收入下降或成本上升所致。"}
```

使用 `prepare_data.py` 会自动转换为标准格式。

---

## 📍 致谢
- [Qwen](https://huggingface.co/Qwen) 模型提供支持
- [HuggingFace Datasets](https://huggingface.co/datasets) 数据接口
- [PEFT](https://github.com/huggingface/peft) LoRA 支持库
