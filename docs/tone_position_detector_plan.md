# Tone Position Detector Plan

Goal: decide, per syllable, whether the selected manual tone should be placed in
`upper_tone` or `lower_tone`.

## Label Source

Use the structured labels generated from:

- `ipa_ocr_work/dataset/shaoxing_structured_tone_labels/structured_tone_syllables.tsv`

Positive class:

- `tone_policy == visual_segment_selected_as_lower`

Negative class:

- `tone_policy == visual_segment_single_tone__lower_null`

Exclude from detector training:

- `tone_position_unknown__lower_null`
- rows without an image
- rows whose crop alignment is still `unmatched`

## Baseline Detector

Use a rule-based connected-component detector as the primary classifier:

1. Convert crop to grayscale.
2. Threshold with Otsu.
3. Estimate baseline from tall IPA components.
4. Find small digit-like components.
5. Split the crop into syllable spans by expected IPA/tone width.
6. Mark a syllable positive if a lower-position component falls in its span.

This is already partially implemented in
`ipa_ocr_work/scripts/build_structured_tone_labels.py` and evaluated by
`ipa_ocr_work/scripts/evaluate_tone_geometry_detector.py`.

Important correction found on 2026-05-24: `lower_only` is a row-level flag and
must not be expanded to "all syllables are lower". Some `lower_only` rows still
contain syllables whose own selected tone is upper/null. The current structured
label builder assigns every syllable from its own visual span whenever the row
image is available.

Current geometry-detector result against the rebuilt structured labels:

- train accuracy/F1: 1.0000 / 1.0000
- val accuracy/F1: 1.0000 / 1.0000
- test accuracy/F1: 1.0000 / 1.0000

This is expected because the label itself is generated from the same explicit
visual rule. The useful outcome is not that a neural classifier learned the
rule, but that the pre-OCR tone-position step is deterministic and auditable.

## Learned Detector

Keep a learned detector only as a robustness check or fallback:

- Input: syllable crop or crop segment.
- Target: `has_lower_tone` binary label.
- Model: small CNN or MobileNet-style encoder.
- Metric: precision/recall/F1 for `has_lower_tone`, reported separately from OCR
  CER.

This model should be cheap. It is a layout classifier, not full OCR. Earlier
CNN/ResNet attempts plateaued around 0.72 test accuracy on noisy syllable crops,
mostly because crop segmentation and row-level labels introduced contradictory
examples.

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
