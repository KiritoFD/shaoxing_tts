from PIL import Image
import numpy as np
import fitz

# Open PDF and get first page at 2x resolution
pdf_path = 'e:/my_pro/shaoxing_123-351.pdf'
doc = fitz.open(pdf_path)
page = doc[0]
mat = fitz.Matrix(2, 2)
pix = page.get_pixmap(matrix=mat)

# Convert to PIL Image
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
img_gray = img.convert('L')

# Crop using the correct 2x coordinates: [[339, 226], [1014, 244], [1012, 319], [337, 301]]
# This is at 2x resolution, so x=337-1012, y=226-319
x1, y1 = 337, 226
x2, y2 = 1012, 319

# Crop the region
cropped = img_gray.crop((x1, y1, x2, y2))
cropped.save('e:/my_pro/moon_ipa_exact.png')

print(f"Cropped image size: {cropped.size}")  # (width, height)
print(f"Cropped image shape: {np.array(cropped).shape}")
