import fitz
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io

pdf_path = r"e:\my_pro\shaoxing_123-351.pdf"

doc = fitz.open(pdf_path)
page = doc[0]
mat = fitz.Matrix(2, 2)

ocr = PaddleOCR(lang='ch')
pix = page.get_pixmap(matrix=mat, alpha=False)
img = Image.open(io.BytesIO(pix.tobytes()))
img_rgb = np.array(img.convert('RGB'))
ocr_result = ocr.predict(img_rgb)
doc.close()

result = ocr_result[0]

print("=== dt_polys (位置坐标) ===")
for i, poly in enumerate(result['dt_polys']):
    print(f"[{i}] {poly.tolist()}")

print("\n=== rec_texts (识别文字) ===")
for i, text in enumerate(result['rec_texts']):
    print(f"[{i}] {text!r}")

print("\n=== rec_scores (置信度) ===")
for i, score in enumerate(result['rec_scores']):
    print(f"[{i}] {score:.4f}")

print("\n=== rec_boxes ===")
print(result['rec_boxes'])

print(f"\n=== 统计 ===")
print(f"总文本框数: {len(result['rec_texts'])}")
y_values = [p[1] for poly in result['dt_polys'] for p in poly]
print(f"Y坐标范围: {min(y_values)} - {max(y_values)} (2x像素)")
print(f"Y坐标范围(1x): {min(y_values)//2} - {max(y_values)//2}")
x_values = [p[0] for poly in result['dt_polys'] for p in poly]
print(f"X坐标范围: {min(x_values)} - {max(x_values)} (2x像素)")
print(f"X坐标范围(1x): {min(x_values)//2} - {max(x_values)//2}")
