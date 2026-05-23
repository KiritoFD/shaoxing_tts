import json
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv

model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'
img_path = 'e:/my_pro/moon_correct.png'

with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r') as f:
    data = json.load(f)
charset = data['scenario']['data']['codec']['charset']

def scale_to_h(img, target_height):
    h, w = img.shape
    f = target_height / h
    if f == 1:
        return img
    w_target = int(round(f * w))
    if w_target == 0:
        w_target = 1
    return cv.resize(img, (w_target, target_height), interpolation=cv.INTER_AREA)

def simple_prep(img, invert=True, transpose=True, pad=16, pad_value=0):
    data = img.astype(np.float32) / 255.0
    if len(data.shape) != 3:
        data = np.expand_dims(data, axis=-1)
    channels = data.shape[-1]
    if data.size > 0:
        if invert:
            data = np.amax(data) - data
    if transpose:
        data = np.swapaxes(data, 1, 0)
    if pad > 0:
        if transpose:
            w = data.shape[1]
            data = np.vstack([np.full((pad, w, channels), pad_value), data, np.full((pad, w, channels), pad_value)])
        else:
            w = data.shape[0]
            data = np.hstack([np.full((w, pad, channels), pad_value), data, np.full((w, pad, channels), pad_value)])
    data = (data * 255).astype(np.uint8)
    if channels == 1:
        data = np.squeeze(data, axis=-1)
    return data

print('Loading model...')
model = tf.saved_model.load(model_path)
infer = model.signatures['serving_default']

print('Loading and preprocessing image...')
img = Image.open(img_path).convert('L')
img_np = np.array(img, dtype=np.uint8)
img_scaled = scale_to_h(img_np, 48)
img_final = simple_prep(img_scaled)
img_final = np.expand_dims(img_final, axis=0)
img_final = np.expand_dims(img_final, axis=-1)
print('Preprocessed image shape:', img_final.shape)

print('Running inference...')
result = infer(img=img_final, img_len=np.array([[img_final.shape[1]]], dtype=np.int32))

print('Decoding CTC output...')
# Important: logits shape is (1, sequence_length, 388), so take result['root'][0, :, :]
logits = result['root'][0, :, :].numpy()  # (sequence_length, 388)
chars = tf.argmax(logits, axis=-1).numpy()  # (sequence_length,)
decoded_norepeat = []
prev = -1
for c in chars:
    if c != prev:
        if 0 < c < len(charset):
            decoded_norepeat.append(charset[c])
    prev = c

result_text = ''.join(decoded_norepeat)

with open('e:/my_pro/final_correct_result.txt', 'w', encoding='utf-8') as f:
    f.write(result_text)

print('Saved result to e:/my_pro/final_correct_result.txt')
