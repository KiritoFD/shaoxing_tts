import json
import tensorflow as tf
import numpy as np

model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'

print('Loading model...')
model = tf.saved_model.load(model_path)
infer = model.signatures['serving_default']

# Create a dummy input
dummy_img = np.zeros((1, 100, 48, 1), dtype=np.uint8)
dummy_len = np.array([[100]], dtype=np.int32)

print('Running inference on dummy input...')
result = infer(img=dummy_img, img_len=dummy_len)

print('\nResult keys:', list(result.keys()))
for key in result:
    tensor = result[key]
    print(f'{key}: shape={tensor.shape}, dtype={tensor.dtype}')
