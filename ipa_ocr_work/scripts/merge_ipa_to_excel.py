import pandas as pd
import json
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
import os
import fitz
from paddleocr import PaddleOCR

# ============ 第一步：提取截图与原始文本的对应关系 ============
print("=" * 60)
print("第一步：提取截图与原始文本的对应关系")
print("=" * 60)

pdf_path = 'e:/my_pro/ipa_ocr_work/data/shaoxing_123-351.pdf'
output_dir = 'e:/my_pro/ipa_ocr_work/images/ipa_v2'

# 使用PyMuPDF渲染图像（2x分辨率）
doc = fitz.open(pdf_path)
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

# 使用PaddleOCR识别渲染后的图像
ocr = PaddleOCR(lang='ch')

print('正在识别文本...')
result = ocr.ocr(img)

def has_non_chinese(text):
    for char in text:
        codepoint = ord(char)
        if not ((0x4e00 <= codepoint <= 0x9fff) or
                (0x3400 <= codepoint <= 0x4dbf) or
                (0x20000 <= codepoint <= 0x2a6df)):
            return True
    return False

# 新版PaddleOCR返回格式是字典
if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
    # 新版格式
    dt_polys = result[0].get('dt_polys', [])
    rec_texts = result[0].get('rec_texts', [])
    rec_scores = result[0].get('rec_scores', [])

    all_boxes = []
    for i, (poly, text, score) in enumerate(zip(dt_polys, rec_texts, rec_scores)):
        all_boxes.append({
            'box': poly.tolist() if hasattr(poly, 'tolist') else poly,
            'text': text,
            'score': score
        })
else:
    # 旧版格式
    all_boxes = []
    for line in result:
        for word in line:
            box = word[0]
            text = word[1][0]
            score = word[1][1]
            all_boxes.append({
                'box': box,
                'text': text,
                'score': score
            })

non_chinese_boxes = []
for i, item in enumerate(all_boxes):
    text = item['text']
    if has_non_chinese(text):
        non_chinese_boxes.append((i, item))

print(f"共识别到 {len(all_boxes)} 个文本框")
print(f"包含非汉字的文本框: {len(non_chinese_boxes)} 个")

# 保存截图
padding = 10
padding_x = 500  # 横向增加更多padding来包含完整的IPA区域
idx_to_info = {}
for idx, (orig_idx, item) in enumerate(non_chinese_boxes):
    box = item['box']
    text = item['text']
    box_array = np.array(box)
    x1 = int(min([point[0] for point in box])) - padding_x
    y1 = int(min([point[1] for point in box])) - padding
    x2 = int(max([point[0] for point in box])) + padding_x
    y2 = int(max([point[1] for point in box])) + padding
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    crop = img[y1:y2, x1:x2]
    filename = f'{output_dir}/ipa_v2_{idx}_orig{orig_idx}.png'
    cv.imwrite(filename, cv.cvtColor(crop, cv.COLOR_RGB2BGR))

    idx_to_info[idx] = {
        'text': text,
        'box': item['box'],
        'orig_idx': orig_idx
    }
    print(f"  {idx}: {text[:30]}...")

# ============ 第二步：IPA识别 ============
print("\n" + "=" * 60)
print("第二步：IPA识别")
print("=" * 60)

with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
charset = data['scenario']['data']['codec']['charset']

TARGET_CHARS = set('p pʰ b m f v t tʰ d n l s sʰ z ɕ ɕʰ ʑ ȵ k kʰ ɡ ŋ h ɦ ɿ a ᴇ ɤ ɒ o æ̃ ẽ ø̃ ə ø ʔ i u y'.split())

CHAR_MAP = {
    'Ɂ': 'ʔ',
    'ɧ̃': 'ɧ',
    'ɧ': 'ɦ',
    'c': 'ɕ',
    'ʈ': 't',
}

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
    result = []
    i = 0
    while i < len(chars):
        char = chars[i]
        if char == '' or char == ' ' or char in FILTER_OUT:
            i += 1
            continue
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
        print(f"Error: {e}")
        return None, None

print('Loading IPA model...')
model = tf.saved_model.load('e:/my_pro/ocr-ipa/model/calamari/best.ckpt')
infer = model.signatures['serving_default']

ipa_results = {}
for idx in idx_to_info:
    img_path = f'{output_dir}/ipa_v2_{idx}_orig{idx_to_info[idx]["orig_idx"]}.png'
    orig_result, filtered_result = recognize_image(img_path)
    ipa_results[idx] = {
        'original_text': idx_to_info[idx]['text'],
        'ipa_raw': orig_result,
        'ipa_filtered': filtered_result
    }
    print(f"  {idx}: {filtered_result}")

# ============ 第三步：匹配Excel ============
print("\n" + "=" * 60)
print("第三步：匹配Excel并合并")
print("=" * 60)

df = pd.read_excel('e:/my_pro/result_all_converted.xlsx')

df_with_page = df[df['页码'].notna()].copy()
print(f"有页码的记录数: {len(df_with_page)}")

# 添加IPA列
df_with_page['IPA识别'] = ''

matched_count = 0
skipped_count = 0

for idx, ipa_info in ipa_results.items():
    orig_text = ipa_info['original_text']
    ipa_filtered = ipa_info['ipa_filtered']

    # 提取起始汉字词组（连续的汉字序列）
    def extract_chinese_words(text):
        words = []
        current_word = ''
        for char in text:
            codepoint = ord(char)
            if (0x4e00 <= codepoint <= 0x9fff) or (0x3400 <= codepoint <= 0x4dbf):
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ''
        if current_word:
            words.append(current_word)
        return words

    chinese_words = extract_chinese_words(orig_text)
    first_word = chinese_words[0] if chinese_words else ''

    if not first_word:
        print(f"  跳过 {idx}: 未找到起始汉字词组")
        skipped_count += 1
        continue

    # 在Excel中查找匹配的汉字词组
    matched = False
    for i, row in df_with_page.iterrows():
        if row['汉字'] == first_word:
            df_with_page.at[i, 'IPA识别'] = ipa_filtered
            print(f"  匹配: 词组'{first_word}' -> IPA: {ipa_filtered}")
            matched = True
            matched_count += 1
            break

    if not matched:
        print(f"  未匹配: 词组'{first_word}', IPA: {ipa_filtered}")
        skipped_count += 1

print(f"\n匹配完成: 成功 {matched_count}, 跳过 {skipped_count}")

df_with_page.to_excel('e:/my_pro/result_all_converted.xlsx', index=False)
print(f"\n结果已保存到 e:/my_pro/result_all_converted.xlsx")
