import json
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
import os

print(f"TensorFlow version: {tf.__version__}")

model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'
images_dir = 'e:/my_pro/ipa_ocr_work/images/ipa_candidates_gpu2'
output_file = 'e:/my_pro/ipa_ocr_work/results/batch_recognition_results.txt'

# 加载模型配置
with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
charset = data['scenario']['data']['codec']['charset']

# 用户指定的目标字符集
TARGET_CHARS = set('p pʰ b m f v t tʰ d n l s sʰ z ɕ ɕʰ ʑ ȵ k kʰ ɡ ŋ h ɦ ɿ a ᴇ ɤ ɒ o æ̃ ẽ ø̃ ə ø ʔ i u y'.split())

# 字符映射：模型字符 -> 用户字符
CHAR_MAP = {
    'Ɂ': 'ʔ',  # 喉塞音变体
    'ɧ̃': 'ɧ',  # ɧ̃去掉鼻化
    'ɧ': 'ɦ',   # ɧ映射为ɦ
    'c': 'ɕ',   # c映射为ɕ
    'ʈ': 't',   # ʈ映射为t
}

# 要过滤掉的字符
FILTER_OUT = {'(', ')', '[', ']', '{', '}', '<', '>', '|', 'ǀ', "'", '’', '.', ',', ';', ':', '!', '?', '̤'}

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

def filter_chars(chars):
    """过滤字符，只保留用户指定的字符"""
    result = []
    i = 0
    while i < len(chars):
        char = chars[i]
        if char == '' or char == ' ' or char in FILTER_OUT:
            i += 1
            continue

        # 循环映射，直到没有对应映射为止（如 ɧ̃ -> ɧ -> ɦ）
        while char in CHAR_MAP:
            char = CHAR_MAP[char]
        if i + 1 < len(chars):
            next_char = chars[i + 1]
            if next_char == 'ʰ':
                combined = char + 'ʰ'
                if combined in TARGET_CHARS:
                    result.append(combined)
                    i += 2
                    continue
            elif next_char == '̃':
                combined = char + '̃'
                if combined in TARGET_CHARS:
                    result.append(combined)
                    i += 2
                    continue
        if char in TARGET_CHARS:
            result.append(char)
        i += 1
    return ''.join(result)

def recognize_image(img_path):
    try:
        img = Image.open(img_path).convert('L')
        img_np = np.array(img, dtype=np.uint8)
        
        _, binary = cv.threshold(img_np, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        img_scaled = scale_to_h(binary, 48)
        img_final = simple_prep(img_scaled)
        img_final = np.expand_dims(img_final, axis=0)
        img_final = np.expand_dims(img_final, axis=-1)
        
        result = infer(img=img_final, img_len=np.array([[img_final.shape[1]]], dtype=np.int32))
        logits = result['root_3'][0, :, :].numpy()
        chars = tf.argmax(logits, axis=-1).numpy()
        
        decoded_all = []
        prev = -1
        for c in chars:
            if c != prev:
                if 0 < c < len(charset):
                    decoded_all.append(charset[c])
            prev = c
        
        filtered = filter_chars(decoded_all)
        return ''.join(decoded_all), filtered
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None

# 加载模型
print('Loading model...')
model = tf.saved_model.load(model_path)
infer = model.signatures['serving_default']

# 获取所有图片文件
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png') and f.startswith('ipa_candidate')])
print(f'Found {len(image_files)} image files')

# 批量识别
results = []
for i, filename in enumerate(image_files):
    img_path = os.path.join(images_dir, filename)
    orig_result, filtered_result = recognize_image(img_path)
    
    if orig_result is not None:
        results.append({
            'filename': filename,
            'original': orig_result,
            'filtered': filtered_result
        })
        print(f'{i+1}/{len(image_files)}: {filename} -> {filtered_result}')
    else:
        results.append({
            'filename': filename,
            'original': 'ERROR',
            'filtered': 'ERROR'
        })
        print(f'{i+1}/{len(image_files)}: {filename} -> ERROR')

# 保存结果
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('=' * 80 + '\n')
    f.write('IPA OCR 批量识别结果\n')
    f.write('=' * 80 + '\n\n')
    
    for r in results:
        f.write(f"文件: {r['filename']}\n")
        f.write(f"原始识别: {r['original']}\n")
        f.write(f"过滤结果: {r['filtered']}\n")
        f.write('-' * 60 + '\n')

print(f'\n结果已保存到: {output_file}')
