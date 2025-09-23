# 请输入标题

## 项目说明
本仓库用于对大模型的训练数据进行混合、sft训练和在LiveCodeBench评测集上进行评测，同时支持快速生成训练和评测所需的配置文件和脚本代码

## 主要功能
1.qwen系列模型在LLaMA-Factory框架上的sft训练和配置文件自动生成
2.qwen系列模型在opencompass上的LiveCodeBench评测和配置文件和脚本文件自动生成

## 目录结构
```
.
├── utils               #自动化数据混合和文件生成工具
├── LLaMA-Factory       #LLaMA-Factory大模型sft代码仓库
├── LLaMA-Factory.yml   #配置LLaMA-Factory运行环境所需的python包及对应版本，仅供参考
├── opencompass         #opencompass大模型评测代码仓库
├── opencompass.yml     #配置opencompass运行环境所需的python包及对应版本，仅供参考
└── README.md           #项目说明文档
```

## 使用说明
### 1.环境配置
推荐方案：根据LLaMA-Factory和opencompass的官方教学文档进行配置
```
#LLaMA-Factory环境配置
conda create --name LLaMA-Factory python=3.10
conda activate LLaMA-Factory
cd LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation

#opencompass环境配置
conda create --name opencompass python=3.10
conda activate opencompass
cd opencompass
pip install -e .
```
备选方案：根据环境配置文件进行配置
```
#LLaMA-Factory环境配置
conda env create -f LLaMA-Factory.yml
#opencompass环境配置
conda env create -f opencompass.yml
#PS：yml文件中对python包的版本限制较死，在安装时可能出现找不到对应版本的问题，故仅供参考
```

### 2.数据集准备
#### 1) sft数据准备

#### 2) LiveCodeBench评测数据准备
因为opencompass目前尚不支持LiveCodeBench数据集的自动下载，所以我们需要手动从Huggingface上下载该数据集
```
cd opencompass
mkdir data
cd data

git lfs install
git clone https://huggingface.co/datasets/livecodebench/code_generation_lite
git clone https://huggingface.co/datasets/livecodebench/execution-v2
git clone https://huggingface.co/datasets/livecodebench/test_generation
#PS: 如遇网络问题，可使用国内镜像站https://hf-mirror.com
```

### 3.自动化工具使用
针对不同使用需求，utils目录下目前有两套文件自动生成的工具
