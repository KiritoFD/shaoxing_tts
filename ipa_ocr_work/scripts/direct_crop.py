import fitz
from PIL import Image
import io

pdf_path = r"e:\my_pro\shaoxing_123-351.pdf"

doc = fitz.open(pdf_path)
page = doc[0]
mat = fitz.Matrix(2, 2)
pix = page.get_pixmap(matrix=mat, alpha=False)
img = Image.open(io.BytesIO(pix.tobytes()))
doc.close()

print(f"图像尺寸: {img.size}")

box = [[339, 226], [1014, 244], [1012, 319], [337, 301]]

y_coords = [p[1] for p in box]
x_coords = [p[0] for p in box]
y1, y2 = min(y_coords), max(y_coords)
x1, x2 = min(x_coords), max(x_coords)

print(f"原始坐标(2x): x=[{x1}, {x2}], y=[{y1}, {y2}]")

x1_1x = x1 // 2
y1_1x = y1 // 2
x2_1x = x2 // 2
y2_1x = y2 // 2

print(f"转换坐标(1x): x=[{x1_1x}, {x2_1x}], y=[{y1_1x}, {y2_1x}]")

pad = 20
x1_crop = max(0, x1_1x - pad)
y1_crop = max(0, y1_1x - pad)
x2_crop = x2_1x + pad
y2_crop = y2_1x + pad

print(f"裁剪坐标(1x): x=[{x1_crop}, {x2_crop}], y=[{y1_crop}, {y2_crop}]")

cropped = img.crop((x1_crop, y1_crop, x2_crop, y2_crop))
cropped.save(r"e:\my_pro\direct_crop.png")
print(f"已保存: e:\my_pro\direct_crop.png")

print(f"\n验证：原坐标2x的y=226-319对应1x应该是113-159")
print(f"我切的是1x的y={y1_crop}-{y2_crop}")
