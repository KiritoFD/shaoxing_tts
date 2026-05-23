import paddle
try:
    paddle.device.set_device('gpu:0')
except:
    pass
from paddleocr import PaddleOCR
import fitz
import cv2 as cv
import numpy as np
import os
import sys
import re
import pandas as pd
import json
import tensorflow as tf
from PIL import Image

print("=" * 60)
print("IPA OCR 全自动处理脚本")
print("=" * 60)

PDF_PATH = 'e:/my_pro/ipa_ocr_work/data/shaoxing_123-351.pdf'
PADDLE_OUTPUT = 'e:/my_pro/ipa_ocr_work/images/ipa_candidates_gpu2'
CALAMARI_MODEL = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'
CHARSET_FILE = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json'
IPA_RESULT_FILE = 'e:/my_pro/ipa_ocr_work/results/batch_recognition_results.txt'
EXCEL_FILE = 'e:/my_pro/result_all_converted.xlsx'

TARGET_CHARS = set('p pʰ b m f v t tʰ d n l s sʰ z ɕ ɕʰ ʑ ȵ k kʰ ɡ ŋ h ɦ ɿ a ᴇ ɤ ɒ o æ̃ ẽ ø̃ ə ø ʔ i u y'.split())
CHAR_MAP = {'Ɂ': 'ʔ', 'ɧ̃': 'ɧ', 'ɧ': 'ɦ', 'c': 'ɕ', 'ʈ': 't'}
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

def process_paddle(page_idx):
    print(f"  PaddleOCR提取第{page_idx+123}页...")
    os.makedirs(PADDLE_OUTPUT, exist_ok=True)

    doc = fitz.open(PDF_PATH)
    page = doc[page_idx]

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    ocr = PaddleOCR(lang='ch')
    result = ocr.ocr(img)

    all_boxes = []
    for line in result:
        for word in line:
            box = word[0]
            text = word[1][0]
            score = word[1][1]
            all_boxes.append({'box': box, 'text': text, 'score': score})

    padding = 5
    for idx, item in enumerate(all_boxes):
        box = item['box']
        x1 = int(min([point[0] for point in box])) - padding
        y1 = int(min([point[1] for point in box])) - padding
        x2 = int(max([point[0] for point in box])) + padding
        y2 = int(max([point[1] for point in box])) + padding
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]
        filename = f'{PADDLE_OUTPUT}/ipa_candidate_{idx}.png'
        cv.imwrite(filename, cv.cvtColor(crop, cv.COLOR_RGB2BGR))

    with open(f'{PADDLE_OUTPUT}/ocr_results.txt', 'w', encoding='utf-8') as f:
        for idx, item in enumerate(all_boxes):
            f.write(f'{idx}|{item["text"]}|{item["score"]:.2f}\n')

    return len(all_boxes)

def process_calamari(num_images):
    print(f"  Calamari OCR识别...")
    charset = json.load(open(CHARSET_FILE, 'r', encoding='utf-8'))['scenario']['data']['codec']['charset']
    model = tf.saved_model.load(CALAMARI_MODEL)
    infer = model.signatures['serving_default']

    results = []
    for i in range(num_images):
        img_path = f'{PADDLE_OUTPUT}/ipa_candidate_{i}.png'
        if not os.path.exists(img_path):
            continue

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
        results.append((f'ipa_candidate_{i}.png', filtered))

    with open(IPA_RESULT_FILE, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('IPA OCR 批量识别结果\n')
        f.write('=' * 80 + '\n\n')
        for filename, ipa in results:
            f.write(f"文件: {filename}\n")
            f.write(f"过滤结果: {ipa}\n")
            f.write('-' * 60 + '\n')

    return results

def merge_to_excel(target_page):
    print(f"  合并到Excel...")
    paddle_results = {}
    with open(f'{PADDLE_OUTPUT}/ocr_results.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                idx = int(parts[0])
                paddle_results[idx] = parts[1]

    ipa_results = {}
    if os.path.exists(IPA_RESULT_FILE):
        with open(IPA_RESULT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            blocks = content.split('------------------------------------------------------------')
            for block in blocks:
                if '文件:' in block and '过滤结果:' in block:
                    file_match = re.search(r'文件:\s*(\S+)', block)
                    filtered_match = re.search(r'过滤结果:\s*(\S*)', block)
                    if file_match and filtered_match:
                        ipa_results[file_match.group(1)] = filtered_match.group(1).strip()

    idx_to_ipa = {}
    for filename, ipa in ipa_results.items():
        match = re.search(r'ipa_candidate_(\d+)\.png', filename)
        if match:
            idx_to_ipa[int(match.group(1))] = ipa

    hanzi_to_ipa = {}
    for idx, text in paddle_results.items():
        if idx in idx_to_ipa:
            hanzi = ''
            for c in text:
                if '\u4e00' <= c <= '\u9fff':
                    hanzi += c
                else:
                    break
            if hanzi:
                hanzi_to_ipa[hanzi] = idx_to_ipa[idx]

    df = pd.read_excel(EXCEL_FILE)
    df_with_page = df[df['页码'].notna()].copy()
    page_data = df_with_page[df_with_page['页码'] == target_page].copy()

    matched_count = 0
    for idx, row in page_data.iterrows():
        hanzi = str(row['汉字']).strip()
        if hanzi in hanzi_to_ipa:
            df_with_page.loc[idx, 'IPA识别'] = hanzi_to_ipa[hanzi]
            matched_count += 1

    df_with_page.to_excel(EXCEL_FILE, index=False)
    print(f"  第{target_page}页: 匹配 {matched_count}/{len(page_data)} 个词条")

def main():
    if len(sys.argv) > 1:
        start_page = int(sys.argv[1])
    else:
        start_page = 123

    if len(sys.argv) > 2:
        end_page = int(sys.argv[2])
    else:
        end_page = 351

    print(f"将处理第 {start_page} 页到第 {end_page} 页")
    print()

    for page in range(start_page, end_page + 1):
        page_idx = page - 123
        print(f"\n处理第{page}页 (索引{page_idx})...")

        try:
            num_boxes = process_paddle(page_idx)
            print(f"  提取了 {num_boxes} 个文本框")

            results = process_calamari(num_boxes)
            print(f"  识别了 {len(results)} 个IPA结果")

            merge_to_excel(page)
        except Exception as e:
            print(f"  错误: {e}")

    print("\n" + "=" * 60)
    print("全部处理完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()
