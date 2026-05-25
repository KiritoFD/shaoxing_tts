# Shaoxing Wu-Pinyin to IPA Rules

This project treats the `wupin` column in `result_all_converted.xlsx` as the
trusted manual label. The legacy IPA/OCR column is not used as ground truth.

The current OCR training label is generated from Wu-pinyin by rule, then checked
by round-trip conversion. The generated IPA keeps ordinary tone digits such as
`52`, not superscript digits, because the OCR target needs a stable machine
label.

## Trusted Page Range

The spreadsheet `page` column is the printed/source page number, not the PDF
page index. In the current PDF export:

- source page `123` is PDF page `1`
- trusted PDF pages `1-136` are source pages `123-258`

The clean training dataset is therefore:

- row manifest: `ipa_ocr_work/dataset/shaoxing_pdf136_clean/ocr_selected_all/eval_manifest.tsv`
- syllable manifest: `ipa_ocr_work/dataset/shaoxing_pdf136_clean/syllable_ocr_all/eval_manifest.tsv`

## Confirmed Wu-Pinyin Inventory

Initials:

```text
p ph b m f v
t th d n l
ts tsh dz s z
c ch j gn sh zh
k kh g ng h gh
```

Finals:

```text
y a e eu au o aen een oen an eon aon on aq aeq eq eoq oeq oq
```

The inventory line that was described as a placeholder is not used as a real
symbol.

## Important Orthography Rules

- `y` is the high rounded vowel final `iu`, IPA `y`.
- `y-` is the simple input spelling for historical/IPA `ghi-`; it maps as
  `ɦi + final`.
- `w-` is the simple input spelling for `ghu`; it maps as `ɦu + final`.
- `ieʔ` is written as `iq` in the manual labels.
- The same checked-tone spelling pattern uses final `q` in labels where a
  source table may write `h`.
- Raw legacy spellings `ghi-`, `ghu`, `ieq`, and `ieh` are canonicalized before
  conversion, but the current trusted clean data has zero raw occurrences.

## Core Initial Mapping

| Wu-pinyin | IPA |
|---|---|
| p ph b m f v | p pʰ b m f v |
| t th d n l | t tʰ d n l |
| ts tsh dz s z | ts tsʰ dz s z |
| c ch j gn sh zh | tɕ tɕʰ dʑ ȵ ɕ ʑ |
| k kh g ng h gh | k kʰ ɡ ŋ h ɦ |
| y- | ɦi + final |
| w- | ɦu + final |

## Core Final Mapping

| Wu-pinyin | IPA |
|---|---|
| y | y |
| a e eu au o | a ᴇ ə ɤ ɒ |
| aen een oen | æ̃ ẽ ø̃ |
| an eon aon on | aŋ əŋ ɒŋ oŋ |
| aq aeq eq eoq oeq oq | aʔ æʔ ᴇʔ əʔ øʔ oʔ |

Medial `i/u/y` combinations are generated compositionally where attested in the
labels, for example `thien -> tʰiẽ`, `kuon -> kuoŋ`, `yaq -> ɦiaʔ`, and
`wo -> ɦuɒ`.

## Whole-Syllable Exceptions

Some high-vowel syllables would double-count the `i/u` glide if split naively,
so they are mapped as whole syllables:

| Wu-pinyin | IPA |
|---|---|
| yi | ɦi |
| yu | ɦy |
| yin | ɦiŋ |
| yiq | ɦiʔ |
| wu | ɦu |
| yoen | ɦiø̃ |
| yoeq | ɦiøʔ |

## Data Checks

Current audit files:

- `ipa_ocr_work/reports/wupin_rule_audit_pdf136.json`
- `ipa_ocr_work/reports/wupin_rule_audit_pdf136.summary.tsv`
- `ipa_ocr_work/reports/wupin_ipa_roundtrip_pdf136_all.summary.tsv`

Current expected headline numbers:

- full CSV rows: `6565`
- full CSV rows with Wu-pinyin: `6535`
- PDF-page-1-to-136 clean row samples: `3773`
- clean syllable samples: `8789`
- row split: train `3288`, val `224`, test `261`
- Wu-pinyin to IPA conversion errors: `0`
- round-trip exact on clean rows: `1.0`
- missing training images in clean row manifest: `0`
- generated labels containing old apical `ɿ`: `0`

Audit caveat:

- The clean manifest still contains attested labels with initials outside the
  compact list above: mostly `mh` in words such as `mha` and `mhau`, plus a small
  number of `tc`.
- These are not conversion failures; the mapping table can convert them using
  the allowed IPA character inventory. They are written to
  `ipa_ocr_work/reports/wupin_rule_audit_pdf136.inventory_review.tsv` for manual
  decision.
- Until the annotator confirms whether these should be rewritten or retained,
  they should be treated as review-risk data rather than proof that the core
  rules are wrong.

## Synthetic Data

Synthetic data is feasible and likely useful, but it should be used as
pretraining or augmentation, not as a replacement for real page crops.

Recommended synthesis:

1. Generate text strings from the trusted Wu-pinyin/IPA lexicon.
2. Render IPA labels with fonts that cover `ɦ ɕ ʑ ȵ ŋ ɒ ᴇ ɤ ø ə æ̃ ẽ ø̃ ʔ`.
3. Apply document-like degradations: blur, low contrast, thresholding,
   scan noise, slight baseline shifts, and crop jitter.
4. Mix real crops and synthetic crops during training, then finish with a
   real-only fine-tune.

The main risk is font mismatch: if the synthetic font shapes differ too much
from the PDF, synthetic data can teach the model the wrong glyph geometry. This
is safest for CTC/row OCR pretraining and for rare syllable coverage, less safe
as the final training distribution.
