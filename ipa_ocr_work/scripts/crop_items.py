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
for i, (text, box, score) in enumerate(page_texts[:10]):
    y_coords = [p[1] for p in box]
    y_center = sum(y_coords) / len(y_coords)
    print(f"[{i}] y={y_center:.0f} (2x) = {y_center//2:.0f} (1x): {text!r}")

print("\n=== 切各个条目 ===")
for i, (text, box, score) in enumerate(page_texts[:5]):
    x_coords = [p[0] for p in box]
    y_coords = [p[1] for p in box]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)

    pad = 20
    x1_p = max(0, x1 - pad)
    y1_p = max(0, y1 - pad)
    x2_p = x2 + pad
    y2_p = y2 + pad

    x1_f = x1_p // 2
    y1_f = y1_p // 2
    x2_f = x2_p // 2
    y2_f = y2_p // 2

    cropped = img.crop((x1_f, y1_f, x2_f, y2_f))
    filename = f"e:\\my_pro\\item_{i}.png"
    cropped.save(filename)

    short_text = text[:15] if len(text) > 15 else text
    print(f"[{i}] y1={y1//2}, y2={y2//2}: {short_text!r} -> {filename}")

print("\n=== 切第0项和第1项合并 ===")
box0 = page_texts[0][1]
box1 = page_texts[1][1]

y1 = min(p[1] for p in box0)
y2 = max(p[1] for p in box1)

pad = 20
y1_p = max(0, y1 - pad)
y2_p = y2 + pad

y1_f = y1_p // 2
y2_f = y2_p // 2

x1_f = 50
x2_f = 1950

cropped = img.crop((x1_f, y1_f, x2_f, y2_f))
cropped.save(r"e:\my_pro\item_0_1.png")
print(f"合并区域: y={y1_f} to {y2_f}, x={x1_f} to {x2_f} -> item_0_1.png")
