import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np
from pprint import pprint
np.random.seed(0)

dummy_input = np.ones([1,10], dtype=np.float32)

# Create a model
i1 = tf.keras.layers.Input(
    shape=[
        dummy_input.shape[1],
        # dummy_input.shape[2],
        # dummy_input.shape[3],
    ],
    batch_size=dummy_input.shape[0],
    dtype=tf.float32,
)
i2 = tf.keras.layers.Input(
    shape=[
        dummy_input.shape[1],
        # dummy_input.shape[2],
        # dummy_input.shape[3],
    ],
    batch_size=dummy_input.shape[0],
    dtype=tf.float32,
)

# o = tf.math.top_k(input=i, k=1, sorted=True)
o = tf.math.multiply(i1, 5)
# o2 = tf.math.multiply(i2, 5)
o = tf.math.multiply(o, i2)
# o = tf.split(value=o, num_or_size_splits=5, axis=1)

model = tf.keras.models.Model(inputs=[i1, i2], outputs=o)
model.summary()
output_path = 'saved_model'
tf.saved_model.save(model, output_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()
open(f"{output_path}/test.tflite", "wb").write(tflite_model)