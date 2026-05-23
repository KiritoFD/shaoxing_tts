import paddle
paddle.device.set_device('gpu:0')
from paddleocr import PaddleOCR
import fitz
import cv2 as cv
import numpy as np
import os
import sys

def process_page(page_num, pdf_path, output_dir):
    print(f"\n{'='*60}")
    print(f"处理第{page_num}页")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    page = doc[page_num]

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    print(f'Image size: {img.shape}')
    print('Running OCR...')

    ocr = PaddleOCR(lang='ch')
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

    print(f'Total: {len(all_boxes)} text boxes')

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

    print(f'Saved {len(all_boxes)} images to {output_dir}')

    cv.imwrite(f'{output_dir}/full_page.png', cv.cvtColor(img, cv.COLOR_RGB2BGR))

    with open(f'{output_dir}/ocr_results.txt', 'w', encoding='utf-8') as f:
        for idx, item in enumerate(all_boxes):
            f.write(f'{idx}|{item["text"]}|{item["score"]:.2f}\n')
    print(f'OCR results saved to {output_dir}/ocr_results.txt')

    return len(all_boxes)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        start_page = int(sys.argv[1])
    else:
        start_page = 123

    if len(sys.argv) > 2:
        end_page = int(sys.argv[2])
    else:
        end_page = 351

    pdf_path = 'e:/my_pro/ipa_ocr_work/data/shaoxing_123-351.pdf'
    output_dir = 'e:/my_pro/ipa_ocr_work/images/ipa_candidates_gpu2'

    for page_num in range(start_page, end_page + 1):
        try:
            process_page(page_num - 123, pdf_path, output_dir)  # PDF索引从0开始
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
