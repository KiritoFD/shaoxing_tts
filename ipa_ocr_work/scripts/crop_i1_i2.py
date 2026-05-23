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
dt_polys = result['dt_polys']
rec_texts = result['rec_texts']
rec_scores = result['rec_scores']

page_texts = list(zip(rec_texts, dt_polys, rec_scores))

print("=== OCR结果 ===")
for i, (text, box, score) in enumerate(page_texts[:5]):
    y_coords = [p[1] for p in box]
    y_center = sum(y_coords) / len(y_coords)
    print(f"[{i}] y={y_center:.0f}(2x)={y_center//2:.0f}(1x): {text!r}")

print("\n=== 切item_1扩大1倍 ===")
text1, box1, score1 = page_texts[1]
x_coords = [p[0] for p in box1]
y_coords = [p[1] for p in box1]
x1, x2 = min(x_coords), max(x_coords)
y1, y2 = min(y_coords), max(y_coords)
h = y2 - y1
pad = h
x1_p = max(0, x1 - pad)
y1_p = max(0, y1 - pad)
x2_p = x2 + pad
y2_p = y2 + pad
x1_f = x1_p // 2
y1_f = y1_p // 2
x2_f = x2_p // 2
y2_f = y2_p // 2
cropped1 = img.crop((x1_f, y1_f, x2_f, y2_f))
cropped1.save(r"e:\my_pro\item_1_big.png")
print(f"item_1: y={y1//2}-{y2//2} -> 扩大后 y={y1_f}-{y2_f}, 尺寸={x2_f-x1_f}x{y2_f-y1_f}")

print("\n=== 切item_2扩大1倍 ===")
text2, box2, score2 = page_texts[2]
x_coords = [p[0] for p in box2]
y_coords = [p[1] for p in box2]
x1, x2 = min(x_coords), max(x_coords)
y1, y2 = min(y_coords), max(y_coords)
h = y2 - y1
pad = h
x1_p = max(0, x1 - pad)
y1_p = max(0, y1 - pad)
x2_p = x2 + pad
y2_p = y2 + pad
x1_f = x1_p // 2
y1_f = y1_p // 2
x2_f = x2_p // 2
y2_f = y2_p // 2
cropped2 = img.crop((x1_f, y1_f, x2_f, y2_f))
cropped2.save(r"e:\my_pro\item_2_big.png")
print(f"item_2: y={y1//2}-{y2//2} -> 扩大后 y={y1_f}-{y2_f}, 尺寸={x2_f-x1_f}x{y2_f-y1_f}")
