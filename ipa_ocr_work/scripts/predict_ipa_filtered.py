import json
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
import sys

print(f"TensorFlow version: {tf.__version__}")

model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'

if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = 'e:/my_pro/ipa_ocr_work/images/moon_correct.png'

with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
charset = data['scenario']['data']['codec']['charset']

# 用户指定的目标字符集
TARGET_CHARS = set('p pʰ b m f v t tʰ d n l s sʰ z ɕ ɕʰ ʑ ȵ k kʰ ɡ ŋ h ɦ ɿ a ᴇ ɤ ɒ o æ̃ ẽ ø̃ ə ø ʔ i u y'.split())

# 模型中存在的字符（用于验证）
AVAILABLE_CHARS = set(charset)

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

# 标记缺失的字符
MISSING_IN_MODEL = TARGET_CHARS - AVAILABLE_CHARS

print('=' * 60)
print('字符集信息:')
print(f'用户指定字符数: {len(TARGET_CHARS)}')
print(f'模型字符集大小: {len(AVAILABLE_CHARS)}')
print(f'模型中不存在的字符: {sorted(MISSING_IN_MODEL)}')
print('=' * 60)

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
    missing_encountered = set()
    
    i = 0
    while i < len(chars):
        char = chars[i]
        
        # 跳过空字符、空格和需要过滤的字符
        if char == '' or char == ' ' or char in FILTER_OUT:
            i += 1
            continue
        
        # 字符映射（循环映射，直到没有对应映射为止）
        while char in CHAR_MAP:
            char = CHAR_MAP[char]
        
        # 尝试组合送气符号
        if i + 1 < len(chars):
            next_char = chars[i + 1]
            # 组合送气
            if next_char == 'ʰ':
                combined = char + 'ʰ'
                if combined in TARGET_CHARS:
                    if combined in AVAILABLE_CHARS or char in AVAILABLE_CHARS:
                        result.append(combined)
                    else:
                        missing_encountered.add(combined)
                    i += 2
                    continue
            # 组合鼻化
            elif next_char == '̃':
                combined = char + '̃'
                if combined in TARGET_CHARS:
                    if combined in AVAILABLE_CHARS or char in AVAILABLE_CHARS:
                        result.append(combined)
                    else:
                        missing_encountered.add(combined)
                    i += 2
                    continue
        
        # 检查是否在目标字符集中
        if char in TARGET_CHARS:
            if char in AVAILABLE_CHARS:
                result.append(char)
            else:
                missing_encountered.add(char)
        
        i += 1
    
    return ''.join(result), missing_encountered

print('Loading model...')
model = tf.saved_model.load(model_path)
infer = model.signatures['serving_default']

print('Loading and preprocessing image...')
img = Image.open(img_path).convert('L')
img_np = np.array(img, dtype=np.uint8)
print(f'Original shape: {img_np.shape}')

_, binary = cv.threshold(img_np, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print(f'After binarization: min={binary.min()}, max={binary.max()}')

img_scaled = scale_to_h(binary, 48)
print(f'After scale_to_h shape: {img_scaled.shape}')

img_final = simple_prep(img_scaled)
print(f'After prep shape: {img_final.shape}')

img_final = np.expand_dims(img_final, axis=0)
img_final = np.expand_dims(img_final, axis=-1)
print(f'Final input shape: {img_final.shape}')

print('Running inference...')
result = infer(img=img_final, img_len=np.array([[img_final.shape[1]]], dtype=np.int32))

print('Decoding CTC output...')
logits = result['root_3'][0, :, :].numpy()
chars = tf.argmax(logits, axis=-1).numpy()

# 解码所有字符
decoded_all = []
prev = -1
for c in chars:
    if c != prev:
        if 0 < c < len(charset):
            decoded_all.append(charset[c])
    prev = c

print(f'\n原始识别结果: {"".join(decoded_all)}')

# 过滤字符
filtered_result, missing_found = filter_chars(decoded_all)
print(f'\n过滤后结果: {filtered_result}')

# 统计使用的字符
used_chars = set(filtered_result)
print(f'\n使用的字符: {sorted(used_chars)}')
print(f'未使用的目标字符: {sorted(TARGET_CHARS - used_chars - MISSING_IN_MODEL)}')

if missing_found:
    print(f'\n⚠️  识别过程中遇到的缺失字符（已跳过）: {sorted(missing_found)}')
else:
    print('\n✅ 识别过程中未遇到缺失字符')

# 保存结果
with open('e:/my_pro/ipa_ocr_work/results/filtered_result.txt', 'w', encoding='utf-8') as f:
    f.write(f'原始识别结果: {{"".join(decoded_all)}}\n')
    f.write(f'过滤后结果: {filtered_result}\n')
    f.write(f'使用的字符: {sorted(used_chars)}\n')
    f.write(f'模型中缺失的目标字符: {sorted(MISSING_IN_MODEL)}\n')

print(f'\n结果已保存到: e:/my_pro/ipa_ocr_work/results/filtered_result.txt')
