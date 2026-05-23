# Wu-Pinyin OCR Training Plan

## Current Target

The manual annotation in `result_all_converted.xlsx` is Wu-pinyin, not IPA. The first OCR model should therefore learn:

```text
printed phonetic notation crop -> Wu-pinyin label from the 拼音 column
```

Then a separate deterministic converter can map Wu-pinyin to IPA. This keeps the OCR problem small and makes labels easy to type and audit.

## Tone Number Rule

The printed PDF has tone numbers in two visual positions:

- lower-right numbers
- upper-right numbers

The annotation rule is:

1. Prefer the lower-right number.
2. If there is no lower-right number, use the upper-right number.

The OCR label writes these tone values as ordinary digits, for example `yoeq1lian55`. The OCR model does not need to preserve whether the number was upper-right or lower-right in the source image.

## Why Not Train Directly on IPA First

Direct IPA training is still possible later, but the current reliable labels are Wu-pinyin. Training on Wu-pinyin first has three advantages:

- smaller and more regular output alphabet
- labels are already available for thousands of rows
- IPA conversion can be tested independently from OCR

The old `IPA识别` column should be treated as legacy OCR/conversion output, not as the primary training label.

## OCR Routes

### 1. Calamari OCR, First Choice

Calamari is a line-level OCR engine. It expects text-line images and matching `.gt.txt` files. This fits our task because the recognizer only needs to read short phonetic crops.

Training data shape:

```text
ipa_ocr_work/dataset/shaoxing_wupin/train/images/page_123_0001.png
ipa_ocr_work/dataset/shaoxing_wupin/train/gt/page_123_0001.gt.txt
```

Pros:

- simple dataset format
- good for small, specialized OCR tasks
- can warmstart from an existing line OCR checkpoint

Risks:

- crops must contain only the printed phonetic notation, not Chinese explanations
- Calamari/TensorFlow needs a dedicated environment

### 2. PaddleOCR Recognition Fine-Tuning

PaddleOCR is useful once the dataset is stable. It supports recognition training with a custom character dictionary and `image<TAB>label` manifests.

Training data shape:

```text
train/images/page_123_0001.png	yoeq1lian55
```

Pros:

- integrated OCR ecosystem
- easier deployment if we keep using PaddleOCR for detection
- custom Wu-pinyin dictionary is straightforward

Risks:

- heavier configuration than Calamari
- current default Python environment does not have `paddle` or `paddleocr`

### 3. TrOCR Fine-Tuning

TrOCR can be tried after a clean dataset exists. It is flexible, but for this project it is a second-stage experiment because a transformer model can overfit noisy small crops.

## Dataset Export Strategy

Use `ipa_ocr_work/scripts/export_wupin_training_data.py`.

It reads:

- PDF: `ipa_ocr_work/data/shaoxing_123-351.pdf`
- Excel labels: `result_all_converted.xlsx`
- label column: `拼音`

It writes:

- Calamari image/ground-truth folders
- Paddle-style `train.txt`, `val.txt`, `test.txt`
- `review.txt` for low-confidence crops
- `charset.txt`
- `manifest.tsv`

The exporter uses the PDF hidden text layer for approximate coordinates, but it separates high-confidence `auto` crops from `review` crops. Only `train/val/test` should be used for first-pass training.

## First Training Commands

Export a sample:

```powershell
py ipa_ocr_work\scripts\export_wupin_training_data.py --start-page 123 --end-page 130 --overwrite
```

Export all pages:

```powershell
py ipa_ocr_work\scripts\export_wupin_training_data.py --start-page 123 --end-page 351 --overwrite
```

Calamari command sketch:

```powershell
calamari-train `
  --trainer.output_dir ipa_ocr_work\models\calamari_wupin `
  --train.images "ipa_ocr_work\dataset\shaoxing_wupin\train\images\*.png" `
  --val.images "ipa_ocr_work\dataset\shaoxing_wupin\val\images\*.png" `
  --trainer.epochs 50 `
  --early_stopping.frequency 1 `
  --early_stopping.n_to_go 8
```

## Verification

Before training:

- visually inspect at least 50 images from `train/images`
- inspect all images listed in `review.txt`
- check that labels are Wu-pinyin, not IPA
- confirm `charset.txt` contains only expected Wu-pinyin letters and digits

After training:

- evaluate CER on `val`
- evaluate full-string match rate on `test`
- inspect errors by tone digits separately from segment letters

## References

- Calamari command line and training: https://calamari-ocr.readthedocs.io/en/latest/doc.command-line-usage.html
- Calamari dataset formats: https://calamari-ocr.readthedocs.io/en/latest/doc.dataset-formats.html
- PaddleOCR recognition training: https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_train/recognition.en.md
- TrOCR documentation: https://huggingface.co/docs/transformers/v4.53.3/en/model_doc/trocr
