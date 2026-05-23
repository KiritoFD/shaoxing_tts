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
print(f'Original shape: {img_np.shape}')

# Crop the width to a reasonable size (e.g., center 800 pixels)
center = img_np.shape[1] // 2
half_width = 400
start = max(0, center - half_width)
end = min(img_np.shape[1], center + half_width)
img_cropped = img_np[:, start:end]
print(f'After cropping: {img_cropped.shape}')

img_scaled = scale_to_h(img_cropped, 48)
print(f'After scale_to_h shape: {img_scaled.shape}')

img_final = simple_prep(img_scaled)
print(f'After prep shape: {img_final.shape}')

# Save preprocessed image for inspection
preprocessed_img = Image.fromarray(img_final)
preprocessed_img.save('e:/my_pro/preprocessed_cropped.png')

img_final = np.expand_dims(img_final, axis=0)
img_final = np.expand_dims(img_final, axis=-1)
print(f'Final input shape: {img_final.shape}')

print('Running inference...')
result = infer(img=img_final, img_len=np.array([[img_final.shape[1]]], dtype=np.int32))

print('Decoding CTC output...')
logits = result['root'][0, :, :].numpy()
chars = tf.argmax(logits, axis=-1).numpy()

# Count character frequencies
unique, counts = np.unique(chars, return_counts=True)
char_freq = sorted(zip(unique, counts), key=lambda x: -x[1])

decoded_norepeat = []
prev = -1
for c in chars:
    if c != prev:
        if 0 < c < len(charset):
            decoded_norepeat.append(charset[c])
    prev = c

result_text = ''.join(decoded_norepeat)

with open('e:/my_pro/result_cropped.txt', 'w', encoding='utf-8') as f:
    f.write(result_text)

print('Saved result to e:/my_pro/result_cropped.txt')
print(f'Top 5 chars:')
for idx, count in char_freq[:5]:
    if idx < len(charset):
        print(f'  {repr(charset[idx])} (index {idx}): {count}')
