# Multi_Agent 多智能体项目

## 项目结构说明（每个文件/文件夹用途）

### 📁 agent/
智能体核心模块，存放所有 Agent 逻辑。
负责：对话管理、任务执行、多智能体协作。

### 📁 models/
大模型调用模块。
封装 LLM（如 OpenAI、DeepSeek 等）的接口与配置。

### 📁 prompts/
提示词模板管理。
统一存放系统提示、任务提示、角色提示等模板。

### 📄 main.py
项目**主入口文件**。
运行整个多智能体系统：
python main.py

### 📄 requirements.txt
项目依赖包列表。
一键安装环境：
pip install -r requirements.txt

### 📄 .env（本地私密文件，不上传 GitHub）
环境变量配置，存放：
- API Key
- 模型配置
- 私密参数

multi_agent/

agent/           智能体核心模块
model/           大模型调用与配置
prompt/          提示词模板管理
main.py          项目主入口
requirements.txt  项目依赖包
.env              本地环境变量（不上传）

