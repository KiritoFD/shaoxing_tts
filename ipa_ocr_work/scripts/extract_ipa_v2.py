import fitz
import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR
import os

ocr = PaddleOCR(lang='ch')

pdf_path = 'e:/my_pro/ipa_ocr_work/data/shaoxing_123-351.pdf'
output_dir = 'e:/my_pro/ipa_ocr_work/images/ipa_v2'

os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)
page = doc[0]
page_rect = page.rect

pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

print('正在识别文本...')
result = ocr.ocr(img)

def has_non_chinese(text):
    """判断文本是否包含非汉字字符"""
    for char in text:
        # 检查是否是非汉字（Unicode范围）
        # 汉字: \u4e00-\u9fff, \u3400-\u4dbf (扩展A), \u20000-\u2a6df (扩展B)
        codepoint = ord(char)
        if not ((0x4e00 <= codepoint <= 0x9fff) or
                (0x3400 <= codepoint <= 0x4dbf) or
                (0x20000 <= codepoint <= 0x2a6df)):
            return True
    return False

def get_all_chars(text):
    """获取文本中的所有字符"""
    chars = []
    for char in text:
        chars.append(char)
    return chars

print(f'页面尺寸: {page_rect}')
print(f'识别到 {len(result[0]) if result else 0} 个文本框')

# 收集所有区域
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

# 筛选包含非汉字字符的区域
non_chinese_boxes = []
for i, item in enumerate(all_boxes):
    text = item['text']
    if has_non_chinese(text):
        non_chinese_boxes.append((i, item))

print(f'\n包含非汉字字符的区域: {len(non_chinese_boxes)} 个')

# 保存所有包含IPA的区域
padding = 3
for idx, (orig_idx, item) in enumerate(non_chinese_boxes):
    box = item['box']
    text = item['text']

    x1 = int(min([point[0] for point in box])) - padding
    y1 = int(min([point[1] for point in box])) - padding
    x2 = int(max([point[0] for point in box])) + padding
    y2 = int(max([point[1] for point in box])) + padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    crop = img[y1:y2, x1:x2]

    filename = f'{output_dir}/ipa_v2_{idx}_orig{orig_idx}.png'
    cv.imwrite(filename, cv.cvtColor(crop, cv.COLOR_RGB2BGR))

    print(f'保存: {filename}')
    print(f'  文本: {repr(text)}')
    print(f'  坐标: {x1},{y1}-{x2},{y2}')
    print()

# 保存整页图像
cv.imwrite(f'{output_dir}/full_page.png', cv.cvtColor(img, cv.COLOR_RGB2BGR))
print(f'整页图像已保存到 {output_dir}/full_page.png')
print(f'共保存 {len(non_chinese_boxes)} 个IPA候选区域')
