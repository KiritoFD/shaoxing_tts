import fitz
from PIL import Image
import io
import numpy as np

pdf_path = r"e:\my_pro\shaoxing_123-351.pdf"

doc = fitz.open(pdf_path)
page = doc[0]
mat = fitz.Matrix(2, 2)
pix = page.get_pixmap(matrix=mat, alpha=False)
img = Image.open(io.BytesIO(pix.tobytes()))
doc.close()

print(f"图像尺寸: {img.size}")  # (宽, 高)

y_center_月亮 = 272 // 2  # 2x to 1x
print(f"'月亮' y_center (1x): {y_center_月亮}")

y_top = 226 // 2  # top of 月亮
print(f"'月亮' y_top (1x): {y_top}")

y_title_est = 150  # 估算"绍兴方言研究"大概位置
print(f"估算标题位置 y ≈ {y_title_est}")

crop = img.crop((100, 50, 600, 250))
crop.save(r"e:\my_pro\debug_page_top.png")
print(f"\n已保存页面上部 debug: e:\\my_pro\\debug_page_top.png")
print(f"裁剪区域: x=100-600, y=50-250")
