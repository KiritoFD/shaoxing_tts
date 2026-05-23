import paddle
from paddleocr import PaddleOCR

print("初始化PaddleOCR (CPU模式)...")
ocr = PaddleOCR(lang='ch', use_gpu=False)
print("PaddleOCR CPU模式初始化成功")

import fitz
import numpy as np

pdf_path = 'e:/my_pro/ipa_ocr_work/data/shaoxing_123-351.pdf'
doc = fitz.open(pdf_path)
page = doc[0]

pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

print("正在识别...")
result = ocr.ocr(img)

print(f"\n识别到 {len(result[0]) if result else 0} 个文本框")
special_count = 0
for line in result:
    for word in line:
        text = word[1][0]
        if any(ord(c) > 127 for c in text):
            special_count += 1
            if special_count <= 20:
                print(f"  文本: {repr(text)}")

print(f"\n共 {special_count} 个包含特殊字符的文本框")
