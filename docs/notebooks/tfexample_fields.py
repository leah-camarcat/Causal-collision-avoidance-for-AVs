import tensorflow as tf

filename = "../../data/motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000"

def print_tf_example_fields(example, indent=0):
    prefix = "  " * indent
    features = example.features.feature
    for key, feature in features.items():
        # 判断 feature 的类型
        if feature.HasField('bytes_list'):
            ftype = "bytes_list"
        elif feature.HasField('float_list'):
            ftype = "float_list"
        elif feature.HasField('int64_list'):
            ftype = "int64_list"
        else:
            ftype = "unknown"
        print(f"{prefix}{key}: {ftype}")

dataset = tf.data.TFRecordDataset(filename, compression_type='')

for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print("TF Example fields:")
    scenario_id = example.features.feature['scenario/id']
    print(scenario_id)
    print_tf_example_fields(example)