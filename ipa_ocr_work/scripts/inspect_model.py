import json
import tensorflow as tf
from PIL import Image
import numpy as np

model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'
img_path = 'e:/my_pro/moon_correct.png'

print('Loading model and inspecting signatures...')
model = tf.saved_model.load(model_path)
print('Model signatures available:', list(model.signatures.keys()))

infer = model.signatures['serving_default']
print('Input signature:', infer.structured_input_signature)
print('Output signature:', infer.structured_output_signature)
