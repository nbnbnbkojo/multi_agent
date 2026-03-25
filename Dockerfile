# 基础镜像：Python 3.10 轻量版，体积小且满足大部分 Python 项目需求
FROM python:3.10-slim

# 设置容器内的工作目录（后续所有操作都在 /app 下执行）
WORKDIR /app

# 第一步：复制依赖文件
COPY requirements.txt .

# 安装依赖：加清华镜像源解决国内下载慢，--no-cache-dir 减小镜像体积
# 替换原来的 pip install 指令，优先用 torch 官方源
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu118
# 第二步：复制整个项目
COPY . .

# （可选）设置环境变量
ENV PORT=8080

# （可选）声明暴露端口（仅声明，运行时需用 -p 映射）
EXPOSE ${PORT}

# 容器启动命令：执行根目录的 main.py（项目入口）
CMD ["python", "main.py"]