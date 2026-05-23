from paddleocr import PaddleOCR
import fitz
import numpy as np
from PIL import Image
import io

# Open the PDF and get first page
pdf_path = 'e:/my_pro/shaoxing_123-351.pdf'
doc = fitz.open(pdf_path)
page = doc[0]

# Convert to image at 2x resolution
mat = fitz.Matrix(2, 2)
pix = page.get_pixmap(matrix=mat)
img_bytes = pix.tobytes("png")
img = Image.open(io.BytesIO(img_bytes))
img.save('e:/my_pro/page_2x.png')

# Run PaddleOCR
ocr = PaddleOCR(lang='ch')
result = ocr.predict(img_bytes)

print("PaddleOCR result structure:")
print(f"Type: {type(result)}")

if isinstance(result, list) and len(result) > 0:
    first_result = result[0]
    print(f"First result type: {type(first_result)}")
    if hasattr(first_result, '__dict__'):
        print(f"First result attributes: {first_result.__dict__.keys()}")
    elif isinstance(first_result, dict):
        print(f"Keys: {first_result.keys()}")

# Try to extract text boxes
all_lines = []
for res in result:
    if hasattr(res, 'dt_polys'):
        for i, poly in enumerate(res.dt_polys):
            text = res.rec_texts[i] if i < len(res.rec_texts) else ''
            score = res.rec_scores[i] if i < len(res.rec_scores) else 0
            all_lines.append((poly, text, score))
    elif isinstance(res, dict):
        if 'dt_polys' in res:
            for i, poly in enumerate(res['dt_polys']):
                text = res['rec_texts'][i] if i < len(res['rec_texts']) else ''
                all_lines.append((poly, text, 0))

# Find "月亮" and nearby IPA text
moon_found = False
for poly, text, score in all_lines:
    if '月亮' in text or '月' in text:
        print(f"\nFound '月亮' at: {poly}")
        moon_found = True

    # Print all text containing IPA-like characters (non-Chinese)
    if any('\u0250' <= c <= '\u02EE' for c in text):
        print(f"Found IPA-like text: {repr(text)} at {poly}")

# Also print all detected text for reference
print("\n\nAll detected text:")
for poly, text, score in all_lines:
    if text.strip():
        print(f"  {repr(text)}: {poly}")
