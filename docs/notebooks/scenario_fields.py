import os
import json
import tensorflow as tf
from google.protobuf.json_format import MessageToDict
from waymo_open_dataset.protos import scenario_pb2

# 假设你下载了一个 Waymo 文件
filename = "../../data/motion_v_1_3_0/uncompressed/scenario/training/training.tfrecord-00000-of-01000"


def extract_scenario_features(filename, num_records=1):
    """
    读取 scenario tfrecord 文件，提取每条 scenario 的所有字段。

    Args:
        filename (str): scenario tfrecord 文件路径
        num_records (int): 读取多少条记录

    Returns:
        list[dict]: 每条 scenario 的字段字典
    """
    dataset = tf.data.TFRecordDataset(filename)
    all_scenarios = []

    for raw_record in dataset.take(num_records):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw_record.numpy())

        # 用 protobuf 的 MessageToDict 递归转换为 dict，包括 repeated fields
        scenario_dict = MessageToDict(scenario, preserving_proto_field_name=True)
        all_scenarios.append(scenario_dict)

    return all_scenarios


# def print_proto(obj, indent=0):
#     prefix = "  " * indent
#     if hasattr(obj, 'DESCRIPTOR'):  # protobuf message
#         for field in obj.DESCRIPTOR.fields:
#             value = getattr(obj, field.name)
#             if field.label == field.LABEL_REPEATED:
#                 print(f"{prefix}{field.name} (repeated): {len(value)} items")
#                 for i, v in enumerate(value):
#                     print(f"{prefix}  - item {i}:")
#                     print_proto(v, indent + 2)
#             else:
#                 if hasattr(value, 'DESCRIPTOR'):  # nested message
#                     print(f"{prefix}{field.name}:")
#                     print_proto(value, indent + 1)
#                 else:
#                     print(f"{prefix}{field.name}: {value}")
#     elif isinstance(obj, (list, tuple)):
#         for i, v in enumerate(obj):
#             print(f"{prefix}- item {i}:")
#             print_proto(v, indent + 1)
#     else:
#         print(f"{prefix}{obj}")
def print_proto(obj, indent=0):
    """
    递归打印 protobuf message 的字段结构（不打印具体数值）。
    """
    prefix = "  " * indent
    if hasattr(obj, 'DESCRIPTOR'):  # protobuf message
        for field in obj.DESCRIPTOR.fields:
            value = getattr(obj, field.name)
            if field.label == field.LABEL_REPEATED:
                # repeated field 只打印类型和数量
                print(f"{prefix}{field.name} (repeated, {len(value)} items)")
                if len(value) > 0 and hasattr(value[0], 'DESCRIPTOR'):
                    # 如果 repeated field 是 nested message，递归打印子字段
                    print_proto(value[0], indent + 1)
            else:
                if hasattr(value, 'DESCRIPTOR'):  # nested message
                    print(f"{prefix}{field.name}:")
                    print_proto(value, indent + 1)
                else:
                    # 普通字段只打印类型
                    print(f"{prefix}{field.name}: {type(value).__name__}")
dataset = tf.data.TFRecordDataset(filename, compression_type='')

# scenarios = extract_scenario_features(filename, num_records=2)
# for i, sc in enumerate(scenarios):
#     print(f"Scenario {i}:")
#     for key, value in sc.items():
#         print(f"  {key}: {type(value)}  -> {value if isinstance(value, (str, int, float)) else 'Complex object/Repeated field'}")
for raw_record in dataset.take(1):  # 这里读取第 1 个 scenario，可以改成 for 循环遍历所有
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(raw_record.numpy())
    print("Scenario fields:")
    print_proto(scenario)