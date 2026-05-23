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

y1_crop = 315
y2_crop = 380
x1_crop = 0
x2_crop = 2000

cropped = img.crop((x1_crop, y1_crop, x2_crop, y2_crop))
cropped.save(r"e:\my_pro\moon_correct.png")
print(f"裁剪坐标: x=[{x1_crop}, {x2_crop}], y=[{y1_crop}, {y2_crop}]")
print(f"尺寸: {x2_crop - x1_crop} x {y2_crop - y1_crop}")
print(f"已保存: e:\my_pro\moon_correct.png")
