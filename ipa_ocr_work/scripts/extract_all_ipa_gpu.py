import fitz
import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR
import os

ocr = PaddleOCR(lang='ch', use_gpu=False)

pdf_path = 'e:/my_pro/ipa_ocr_work/data/shaoxing_123-351.pdf'
output_dir = 'e:/my_pro/ipa_ocr_work/images/ipa_candidates_gpu'

os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)
page = doc[0]

pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

print(f'Image size: {img.shape}')

print('Running OCR...')
result = ocr.ocr(img)

all_boxes = []
for line in result:
    for word in line:
        box = word[0]
        text = word[1][0]
        score = word[1][1]
        all_boxes.append({
            'box': box,
            'text': text,
            'score': score
        })
        print(f'Box {len(all_boxes)}: score={score:.2f}')

print(f'\nTotal: {len(all_boxes)} text boxes')

padding = 5
for idx, item in enumerate(all_boxes):
    box = item['box']
    text = item['text']

    x1 = int(min([point[0] for point in box])) - padding
    y1 = int(min([point[1] for point in box])) - padding
    x2 = int(max([point[0] for point in box])) + padding
    y2 = int(max([point[1] for point in box])) + padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    crop = img[y1:y2, x1:x2]

    filename = f'{output_dir}/ipa_candidate_{idx}.png'
    cv.imwrite(filename, cv.cvtColor(crop, cv.COLOR_RGB2BGR))
    print(f'Saved: {filename} ({x1},{y1}-{x2},{y2})')

cv.imwrite(f'{output_dir}/full_page.png', cv.cvtColor(img, cv.COLOR_RGB2BGR))
print(f'\nFull page saved to {output_dir}/full_page.png')

with open(f'{output_dir}/ocr_results.txt', 'w', encoding='utf-8') as f:
    for idx, item in enumerate(all_boxes):
        f.write(f'{idx}|{item["text"]}|{item["score"]:.2f}\n')
print(f'OCR results saved to {output_dir}/ocr_results.txt')
