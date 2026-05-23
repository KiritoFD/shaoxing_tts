import fitz
import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR
import os

# 初始化PaddleOCR
ocr = PaddleOCR(lang='ch')

# PDF路径
pdf_path = 'e:/my_pro/ipa_ocr_work/data/shaoxing_123-351.pdf'
output_dir = 'e:/my_pro/ipa_ocr_work/images/ipa_candidates'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载PDF第一页
doc = fitz.open(pdf_path)
page = doc[0]

# 获取页面尺寸
page_rect = page.rect
print(f'页面尺寸: {page_rect}')

# 渲染页面为图像（2x分辨率）
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

# 使用PaddleOCR识别
print('正在识别文本...')
result = ocr.ocr(img)

# 保存所有识别结果
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
        print(f'文本: {repr(text):<20} 置信度: {score:.2f}')

print(f'\n共识别到 {len(all_boxes)} 个文本框')

# 定义IPA字符模式（简单检测）
ipa_chars = set('p pʰ b m f v t tʰ d n l s sʰ z ɕ ɕʰ ʑ ȵ k kʰ ɡ ŋ h ɦ ɿ a ᴇ ɤ ɒ o æ̃ ẽ ø̃ ə ø ʔ i u y'.split())

# 筛选可能包含IPA的文本框
ipa_candidates = []
for i, item in enumerate(all_boxes):
    text = item['text']
    # 检查是否包含特殊字符（可能是IPA）
    has_special = any(ord(c) > 127 for c in text)
    if has_special:
        ipa_candidates.append((i, item))
        print(f'候选 {i}: {repr(text)}')

print(f'\n找到 {len(ipa_candidates)} 个IPA候选区域')

# 截取并保存每个候选区域
padding = 5  # 额外padding
for idx, (orig_idx, item) in enumerate(ipa_candidates):
    box = item['box']
    text = item['text']
    
    # 获取坐标
    x1 = int(min([point[0] for point in box])) - padding
    y1 = int(min([point[1] for point in box])) - padding
    x2 = int(max([point[0] for point in box])) + padding
    y2 = int(max([point[1] for point in box])) + padding
    
    # 确保坐标在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    
    # 截取区域
    crop = img[y1:y2, x1:x2]
    
    # 保存
    filename = f'{output_dir}/ipa_candidate_{idx}_orig{orig_idx}.png'
    cv.imwrite(filename, cv.cvtColor(crop, cv.COLOR_RGB2BGR))
    print(f'保存: {filename} (坐标: {x1},{y1}-{x2},{y2}, 文本: {repr(text)})')

print(f'\n所有候选区域已保存到 {output_dir}')

# 同时保存整页图像供参考
cv.imwrite(f'{output_dir}/full_page.png', cv.cvtColor(img, cv.COLOR_RGB2BGR))
print('整页图像已保存')
