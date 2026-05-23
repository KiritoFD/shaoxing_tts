import json
import tensorflow as tf
from PIL import Image
import numpy as np

model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'

print('Loading model...')
model = tf.saved_model.load(model_path)

print('\n--- Model Signatures ---')
print('Available signatures:', list(model.signatures.keys()))

infer = model.signatures['serving_default']

print('\n--- Serving Default Signature Details ---')
print('Inputs:')
for key, tensor_spec in infer.structured_input_signature[1].items():
    print(f'  {key}: shape={tensor_spec.shape}, dtype={tensor_spec.dtype}')

print('\nOutputs:')
print(f'  Return type is a dict: {type(infer.outputs)}')
for key in infer.outputs:
    print(f'  {key}: {infer.outputs[key]}')
