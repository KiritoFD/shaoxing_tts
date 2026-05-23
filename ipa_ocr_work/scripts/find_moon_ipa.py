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

print("=== 完整OCR结果 ===")
for i, (text, box, score) in enumerate(page_texts):
    y_coords = [p[1] for p in box]
    y_center = sum(y_coords) / len(y_coords)
    print(f"[{i}] y_center={y_center:.0f} (2x) {text!r}")

print("\n=== 查找'月亮' ===")
for i, (text, box, score) in enumerate(page_texts):
    if "月亮" in text and "hielia" in text:
        print(f"目标: [{i}] {text!r}")

        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        width = x2 - x1
        height = y2 - y1
        print(f"原始尺寸 (2x): 宽={width}, 高={height}")
        print(f"原始坐标 (2x): x=[{x1}, {x2}], y=[{y1}, {y2}]")

        pad = 30
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = x2 + pad
        y2_pad = y2 + pad

        x1_final = x1_pad // 2
        y1_final = y1_pad // 2
        x2_final = x2_pad // 2
        y2_final = y2_pad // 2

        cropped = img.crop((x1_final, y1_final, x2_final, y2_final))
        cropped.save(r"e:\my_pro\moon_ipa_region.png")
        print(f"\n裁剪区域 (1x): x1={x1_final}, y1={y1_final}, x2={x2_final}, y2={y2_final}")
        print(f"尺寸: {x2_final - x1_final} x {y2_final - y1_final}")
        print(f"已保存到: e:\\my_pro\\moon_ipa_region.png")
        break
