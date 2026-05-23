import json
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2 as cv

def generate_text_image_simple(text, font_path=None, font_size=30, padding=10, fixed_height=48):
    # Simple version, using default font if needed
    if font_path is None:
        font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

    left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    img_width = text_width + 2 * padding

    if fixed_height is not None:
        img_height = fixed_height
    else:
        img_height = text_height + 2 * padding

    img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = padding - left
    if fixed_height is not None:
        y = (img_height - text_height) // 2 - top
    else:
        y = padding - top
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return img.convert("L")

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

# Test with known text
test_text = "hielia, nilian"
print(f'Generating test image with text: {repr(test_text)}')
test_img_pil = generate_text_image_simple(test_text)
test_img_np = np.array(test_img_pil, dtype=np.uint8)
test_img_path = 'e:/my_pro/test_generated.png'
test_img_pil.save(test_img_path)
print(f'Saved test image to {test_img_path}')

# Now predict
model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'
with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r') as f:
    data = json.load(f)
charset = data['scenario']['data']['codec']['charset']

print('Loading model...')
model = tf.saved_model.load(model_path)
infer = model.signatures['serving_default']

print('Preprocessing test image...')
img_scaled = scale_to_h(test_img_np, 48)
img_final = simple_prep(img_scaled)
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

print(f'Original: {repr(test_text)}')
print(f'Recognized: {repr(result_text)}')

with open('e:/my_pro/test_result.txt', 'w', encoding='utf-8') as f:
    f.write(result_text)
