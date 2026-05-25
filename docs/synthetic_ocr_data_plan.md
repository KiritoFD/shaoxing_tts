# Synthetic OCR Data Plan

Synthetic data is feasible for this project. The best use is to improve OCR
pretraining and rare-symbol coverage, then fine-tune on real PDF crops.

## Why It Can Help

The real dataset is small:

- row-level clean samples: `3773`
- syllable-level clean samples: `8789`
- many row labels are nearly unique
- important symbols such as `ɦ`, `ȵ`, `ʑ`, nasal vowels, and `ʔ` are sparse

Synthetic rendering can expose the model to many more combinations of the same
IPA inventory before it sees the noisy scan crops.

## What To Generate

Generate two synthetic sets:

1. Syllable crops
   - Label unit: one IPA syllable plus tone digits.
   - Purpose: closed-set/diagnostic classifier and rare-symbol practice.

2. Row crops
   - Label unit: concatenated IPA+digits string.
   - Purpose: CTC/SVTR row OCR pretraining.

Text source should come from trusted Wu-pinyin labels converted by
`wupin_ipa_convert.py`, plus recombinations sampled from the same syllable
inventory.

## Rendering Requirements

Fonts must cover:

```text
p pʰ b m f v t tʰ d n l s z ts tsʰ dz
ɕ ɕʰ ʑ ȵ k kʰ ɡ ŋ h ɦ
a ᴇ ɤ ɒ o æ̃ ẽ ø̃ ə ø ʔ i u y
0 1 2 3 4 5
```

Candidate Windows fonts should be tested before use. If local fonts fail, use
Noto Serif/Sans or Charis SIL on the Linux server.

## Degradations

Synthetic images should not be clean digital text only. Apply random:

- grayscale contrast changes
- Gaussian blur
- slight erosion/dilation
- scan noise and JPEG artifacts
- threshold/binarization variants
- crop margin jitter
- vertical baseline jitter
- mild horizontal compression/stretching

The goal is to mimic the current PDF: low contrast, uneven blur, and tight row
crops.

## Training Schedule

Recommended order:

1. Pretrain CTC/SVTR on synthetic rows.
2. Mix synthetic and real rows at a low synthetic ratio.
3. Finish with real-only fine-tuning.
4. Score only on the fixed real test split.

Do not report synthetic validation as real OCR accuracy.

## Risks

- Wrong font shapes can hurt final OCR.
- Synthetic tone placement may not match the scanned PDF if rendered as normal
  baseline digits. For this project the current labels use plain digits, so the
  synthetic target should also use plain digits unless the visual target is
  changed.
- Synthetic data will not fix bad crop alignment; it only helps recognition.

## Recommendation

Do it, but as a controlled experiment:

- Start with `50k` synthetic syllables and `50k` synthetic rows.
- Run CTC/SVTR pretraining for a short schedule on the 5090.
- Fine-tune on `shaoxing_pdf136_clean`.
- Compare against the no-synthetic run using the same `test` split.
