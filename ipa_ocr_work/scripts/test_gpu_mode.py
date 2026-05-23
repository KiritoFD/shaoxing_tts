import paddle
paddle.device.set_device('gpu:0')
from paddleocr import PaddleOCR

print("Initializing PaddleOCR (GPU mode)...")
ocr = PaddleOCR(lang='ch')
print("PaddleOCR GPU mode initialized successfully")

import fitz
import numpy as np

pdf_path = 'e:/my_pro/ipa_ocr_work/data/shaoxing_123-351.pdf'
doc = fitz.open(pdf_path)
page = doc[0]

pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

print("Running OCR...")
result = ocr.ocr(img)

print(f"\nDetected {len(result[0]) if result else 0} text boxes")
