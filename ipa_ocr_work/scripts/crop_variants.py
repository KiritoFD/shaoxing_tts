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

print("=== 查找'月亮hielia' ===")
for i, (text, box, score) in enumerate(page_texts):
    if "月亮" in text and "hielia" in text:
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        print(f"目标文字: {text!r}")
        print(f"原始坐标 (2x): x=[{x1}, {x2}], y=[{y1}, {y2}]")

        variants = [
            ("A_紧', padding=5", 5),
            ("B_小', padding=15", 15),
            ("C_中', padding=30", 30),
            ("D_大', padding=50", 50),
            ("E_更大', padding=80", 80),
        ]

        for name, pad in variants:
            x1_p = max(0, x1 - pad)
            y1_p = max(0, y1 - pad)
            x2_p = x2 + pad
            y2_p = y2 + pad

            x1_f = x1_p // 2
            y1_f = y1_p // 2
            x2_f = x2_p // 2
            y2_f = y2_p // 2

            cropped = img.crop((x1_f, y1_f, x2_f, y2_f))
            filename = f"e:\\my_pro\\moon_ipa_{name.replace(' ', '_').replace(',', '')}.png"
            cropped.save(filename)
            print(f"\n{name}")
            print(f"  区域 (1x): x1={x1_f}, y1={y1_f}, x2={x2_f}, y2={y2_f}")
            print(f"  尺寸: {x2_f - x1_f} x {y2_f - y1_f}")
            print(f"  保存: {filename}")
        break

print("\n=== 同时保存整页供参考 ===")
img.save(r"e:\my_pro\full_page_1.png")
print(f"整页保存到: e:\\my_pro\\full_page_1.png")
