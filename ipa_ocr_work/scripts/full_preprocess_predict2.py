import json
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
from pathlib import Path

model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'
img_path = 'e:/my_pro/moon_correct.png'

# Load charset
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

def center_normalize(img, line_height=48, extra_params=(4, 1.0, 0.3)):
    range_val, smoothness, extra = extra_params
    target_height = line_height
    scaled = scale_to_h(img, target_height)
    h, w = scaled.shape
    temp = (scaled / 255).astype(np.float32)
    temp = np.amax(temp) - temp
    amax = np.amax(temp)
    if amax == 0:
        return scaled
    inverted = temp * 1.0 / amax
    h_meas, w_meas = inverted.shape
    smoothed = cv.GaussianBlur(inverted, (0, 0), sigmaX=h_meas * smoothness, sigmaY=h_meas * 0.5, borderType=cv.BORDER_CONSTANT)
    smoothed += 0.001 * cv.blur(smoothed, (w_meas, int(h_meas * 0.5)), borderType=cv.BORDER_CONSTANT)
    a = np.argmax(smoothed, axis=0).astype(np.uint16)
    kernel = cv.getGaussianKernel(int((8.0 * h_meas * extra) + 1), h_meas * extra)
    center = cv.filter2D(a, cv.CV_16U, kernel, borderType=cv.BORDER_REFLECT).flatten()
    deltas = abs(np.arange(h_meas)[:, np.newaxis] - center[np.newaxis, :])
    mad = np.mean(deltas[scaled != 0])
    r = int(1 + range_val * mad)
    hpad = r
    padded = cv.copyMakeBorder(scaled, hpad, hpad, 0, 0, cv.BORDER_CONSTANT, value=0)
    center = center + hpad - r
    new_h = 2 * r
    dewarped = [padded[c : c + new_h, i] for i, c in enumerate(center)]
    dewarped = np.swapaxes(np.array(dewarped, dtype=np.uint8), 1, 0)
    return scale_to_h(dewarped, target_height)

def final_prep(img, normalize=True, invert=True, transpose=True, pad=16, pad_value=0):
    data = img.astype(np.float32) / 255.0
    if len(data.shape) != 3:
        data = np.expand_dims(data, axis=-1)
    channels = data.shape[-1]
    if data.size > 0:
        if normalize:
            amax = np.amax(data)
            if amax > 0:
                data = data * 1.0 / amax
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
img_center_normalized = center_normalize(img_np, line_height=48)
img_final = final_prep(img_center_normalized)
img_final = np.expand_dims(img_final, axis=0)
img_final = np.expand_dims(img_final, axis=-1)
print('Preprocessed image shape:', img_final.shape)

print('Running inference...')
result = infer(img=img_final, img_len=np.array([[img_final.shape[1]]], dtype=np.int32))

print('Decoding CTC output...')
logits = result['root'][0].numpy()
chars = tf.argmax(logits, axis=-1).numpy()
decoded_norepeat = []
prev = -1
for c in chars:
    if c != prev:
        if 0 < c < len(charset):
            decoded_norepeat.append(charset[c])
    prev = c

result_text = ''.join(decoded_norepeat)

with open('e:/my_pro/final_result.txt', 'w', encoding='utf-8') as f:
    f.write(result_text)

print('Saved result to e:/my_pro/final_result.txt')
print('Result written successfully')
