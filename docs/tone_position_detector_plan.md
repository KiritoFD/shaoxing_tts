# Tone Position Detector Plan

Goal: decide, per syllable, whether the selected manual tone should be placed in
`upper_tone` or `lower_tone`.

## Label Source

Use the structured labels generated from:

- `ipa_ocr_work/dataset/shaoxing_structured_tone_labels/structured_tone_syllables.tsv`

Positive class:

- `tone_policy == visual_segment_selected_as_lower`
- `tone_policy == selected_as_lower`

Negative class:

- `tone_policy == visual_segment_single_tone__lower_null`
- `tone_policy == single_tone__lower_null`

Exclude from detector training:

- `tone_position_unknown__lower_null`
- rows without an image
- rows whose crop alignment is still `unmatched`

## Baseline Detector

Start with a rule-based connected-component detector:

1. Convert crop to grayscale.
2. Threshold with Otsu.
3. Estimate baseline from tall IPA components.
4. Find small digit-like components.
5. Split the crop into syllable spans by expected IPA/tone width.
6. Mark a syllable positive if a lower-position component falls in its span.

This is already partially implemented in
`ipa_ocr_work/scripts/build_structured_tone_labels.py`.

## Learned Detector

If the rule detector is not good enough, train a small classifier:

- Input: syllable crop or crop segment.
- Target: `has_lower_tone` binary label.
- Model: small CNN or MobileNet-style encoder.
- Metric: precision/recall/F1 for `has_lower_tone`, reported separately from OCR
  CER.

This model should be cheap. It is a layout classifier, not full OCR.

## Expected Performance

This should be easier than OCR because it ignores IPA character identity. It
only needs to decide whether there is a small dark component below the baseline
near the syllable's tone position.

Likely failure cases:

- Crops include explanation Chinese characters.
- Syllable segmentation is wrong for long multi-syllable rows.
- Blurry lower digits merge with the main IPA glyph.
- Row-level `both_upper_lower` was triggered by a neighboring syllable.

## Recommended Pipeline

Use two stages first:

1. OCR model predicts `ipa_base + selected_tone`.
2. Tone-position detector predicts whether each selected tone goes to
   `upper_tone` or `lower_tone`.

This should outperform a single plain CTC sequence model for the final
structured output, because the two subproblems have different visual cues.
