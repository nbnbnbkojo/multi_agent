import os
import logging
from typing import Optional

from pymilvus import connections, utility, MilvusException, Collection
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# -------------------------- Milvus 核心配置 --------------------------
# Milvus 连接信息（从环境变量读取，提供默认值）
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "food_safety_kb")  # milvus集合名称

# -------------------------- Milvus 连接类 --------------------------
class MilvusConnector:
    """极简版 Milvus 向量数据库连接类（仅负责连接/断开/集合校验）"""

    def __init__(self):
        # 连接状态标识
        self.connected = False
        # Milvus 连接配置
        self.connection_alias = "default"
        self.milvus_host = MILVUS_HOST
        self.milvus_port = MILVUS_PORT
        self.collection_name = COLLECTION_NAME
        # 集合对象（连接成功后绑定）
        self.collection: Optional[Collection] = None
        # 日志配置
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def connect(self) -> tuple[bool, str]:
        """
        建立 Milvus 连接并校验集合是否存在
        :return: (连接状态, 提示信息)
        """
        # 已连接则直接返回成功
        if self.connected:
            self.logger.info("Milvus 已处于连接状态，无需重复连接")
            return True, "✅ 已连接到 Milvus"

        try:
            # 1. 建立 Milvus 服务连接
            connections.connect(
                alias=self.connection_alias,
                host=self.milvus_host,
                port=self.milvus_port,
                timeout=10  # 连接超时时间（秒）
            )
            self.logger.info(f"成功连接到 Milvus 服务: {self.milvus_host}:{self.milvus_port}")

            # 2. 校验目标集合是否存在
            if not utility.has_collection(self.collection_name):
                self.disconnect()  # 连接成功但集合不存在，断开连接
                err_msg = f"❌ 集合 {self.collection_name} 不存在"
                self.logger.error(err_msg)
                return False, err_msg

            # 3. 绑定集合并加载到内存
            self.collection = Collection(self.collection_name)
            self.collection.load()
            self.connected = True

            success_msg = (
                f"✅ Milvus 连接成功 | 集合: {self.collection_name} "
                f"| 数据量: {self.collection.num_entities}"
            )
            self.logger.info(success_msg)
            return True, success_msg

        except MilvusException as e:
            err_msg = f"❌ Milvus 连接异常: {e.message} (错误码: {e.code})"
            self.logger.error(err_msg)
            return False, err_msg
        except Exception as e:
            err_msg = f"❌ Milvus 连接失败: {str(e)}"
            self.logger.error(err_msg)
            return False, err_msg

    def disconnect(self) -> tuple[bool, str]:
        """
        断开 Milvus 连接并释放资源
        :return: (断开状态, 提示信息)
        """
        if not self.connected:
            self.logger.info("Milvus 未建立连接，无需断开")
            return True, "✅ 未建立 Milvus 连接，无需断开"

        try:
            # 1. 释放集合内存（可选但推荐）
            if self.collection:
                self.collection.release()
                self.collection = None

            # 2. 断开连接
            connections.disconnect(self.connection_alias)
            self.connected = False

            success_msg = "✅ Milvus 连接已断开"
            self.logger.info(success_msg)
            return True, success_msg

        except Exception as e:
            err_msg = f"❌ 断开 Milvus 连接失败: {str(e)}"
            self.logger.error(err_msg)
            return False, err_msg

    def __del__(self):
        """析构函数：对象销毁时自动断开连接"""
        if self.connected:
            self.disconnect()

