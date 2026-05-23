import fitz
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import io

pdf_path = r"e:\my_pro\shaoxing_123-351.pdf"

doc = fitz.open(pdf_path)
print(f"PDF页数: {len(doc)}")

page = doc[0]
print(f"当前页面编号: {page.number}")
print(f"页面尺寸: {page.rect}")

mat = fitz.Matrix(2, 2)

ocr = PaddleOCR(lang='ch')
pix = page.get_pixmap(matrix=mat, alpha=False)
img = Image.open(io.BytesIO(pix.tobytes()))
print(f"图像尺寸: {img.size}")

img_rgb = np.array(img.convert('RGB'))
ocr_result = ocr.predict(img_rgb)
doc.close()

result = ocr_result[0]
dt_polys = result['dt_polys']
rec_texts = result['rec_texts']
rec_scores = result['rec_scores']

page_texts = list(zip(rec_texts, dt_polys, rec_scores))

print("\n=== OCR结果 ===")
for i, (text, box, score) in enumerate(page_texts):
    y_coords = [p[1] for p in box]
    y_center = sum(y_coords) / len(y_coords)
    x_coords = [p[0] for p in box]
    x_center = sum(x_coords) / len(x_coords)
    print(f"[{i}] y={y_center:.0f}(2x)={y_center//2:.0f}(1x), x={x_center:.0f}(2x)={x_center//2:.0f}(1x): {text!r}")

print("\n=== 扩大2倍切第0项 ===")
text0, box0, score0 = page_texts[0]
x_coords = [p[0] for p in box0]
y_coords = [p[1] for p in box0]
x1, x2 = min(x_coords), max(x_coords)
y1, y2 = min(y_coords), max(y_coords)

w = x2 - x1
h = y2 - y1
print(f"原始尺寸: {w} x {h}")

pad = h  # 扩大1倍
x1_p = max(0, x1 - pad)
y1_p = max(0, y1 - pad)
x2_p = x2 + pad
y2_p = y2 + pad

x1_f = x1_p // 2
y1_f = y1_p // 2
x2_f = x2_p // 2
y2_f = y2_p // 2

cropped = img.crop((x1_f, y1_f, x2_f, y2_f))
cropped.save(r"e:\my_pro\item_0_big.png")
print(f"扩大区域(1x): x1={x1_f}, y1={y1_f}, x2={x2_f}, y2={y2_f}")
print(f"尺寸: {x2_f - x1_f} x {y2_f - y1_f}")
print(f"保存: item_0_big.png")

print("\n=== 切页面顶部区域(0-500 y, 2x) ===")
cropped_top = img.crop((0, 0, 2000, 500))
cropped_top.save(r"e:\my_pro\page_top.png")
print(f"保存: page_top.png (0-250 in 1x)")

print("\n=== 切页面中上部区域(200-600 y, 2x) ===")
cropped_mid = img.crop((0, 200, 2000, 600))
cropped_mid.save(r"e:\my_pro\page_mid.png")
print(f"保存: page_mid.png (100-300 in 1x)")
