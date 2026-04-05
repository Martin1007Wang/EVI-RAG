import lmdb
import torch
from safetensors.torch import load
from pathlib import Path

# 检查LMDB文件中的字段名
embeddings_dir = Path("/mnt/data/retrieval/webqsp/materialized/embeddings")
lmdb_path = embeddings_dir / "train.lmdb"

print(f"检查LMDB文件: {lmdb_path}")

try:
    env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=1)

    with env.begin() as txn:
        # 获取第一个键值对
        cursor = txn.cursor()
        if cursor.first():
            key = cursor.key()
            value = cursor.value()

            print(f"第一个样本的键: {key.decode('utf-8')}")

            # 解析样本数据
            data = load(value)
            print(f"字段名: {list(data.keys())}")

            # 检查是否有a_local_indicies字段
            for field in data.keys():
                if "local" in field.lower() and "indice" in field.lower():
                    print(f"找到本地索引字段: {field}")
                    print(f"  类型: {type(data[field])}")
                    print(f"  形状: {data[field].shape}")
                    print(f"  内容: {data[field].tolist()}")

        else:
            print("LMDB文件为空")

    env.close()

except Exception as e:
    print(f"读取LMDB文件出错: {e}")
    import traceback

    traceback.print_exc()
