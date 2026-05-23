# Structured Tone Label Schema

This dataset normalizes Shaoxing phonetic labels into per-syllable tone slots.

## Observed Rules

The trusted source is still the manual Wu-pinyin column in
`result_all_converted.clean.csv` / `result_all_converted.with_ipa.csv`.

Tone placement in the source PDF follows two visual cases:

- A syllable may have only one visible tone number, normally at the upper-right
  position.
- A syllable may have both an upper/right number and a lower/right number.

The annotator's rule, confirmed by spot checks against the PDF, is:

- If a lower/right number exists for that syllable, the manual Wu-pinyin tone is
  the lower/right number.
- If no lower/right number exists, the manual Wu-pinyin tone is the upper/right
  number.

Examples:

- `月亮 yoeq1lian55`: the first syllable has an upper visual value and a lower
  `1`; the label uses `1`.
- `扁担星 pien335taen55shin52`: `taen55` uses the lower/right `55`; `pien335`
  and `shin52` are single visible tone cases and use the upper/right values.
- `七簇星 chiq3tshoq5shin52`: checked syllables can use lower/right single
  digits such as `3` and `5`, while another syllable in the same crop can still
  use a single upper/right `52`.

Important detail: `both_upper_lower` was first detected at crop/row level. That
does not mean every syllable in the crop has two tone numbers. The structured
version therefore assigns `upper_tone` / `lower_tone` per syllable, not per row.

Each syllable has:

- `ipa_base`: IPA without tone digits.
- `selected_tone`: the trusted tone digits from the manual Wu-pinyin label.
- `upper_tone`: the selected tone when it is visually treated as the upper/right tone slot.
- `lower_tone`: the selected tone when it is visually treated as the lower/right tone slot.
- `tone_policy`: how the slot assignment was made.

## Slot Convention

- If a syllable has only one visible tone number, store it in `upper_tone` and set `lower_tone` to null.
- If a syllable has a lower/right tone number selected by the annotation rule, store it in `lower_tone`.
- For rows with both upper and lower tone numbers, per-syllable slot assignment is currently image-heuristic: the crop is split left-to-right by expected syllable width, and lower-position tone components decide whether `lower_tone` is populated.

Current `tone_policy` values:

- `single_tone__lower_null`: a single visible tone case; selected tone is put in
  `upper_tone`.
- `selected_as_lower`: row-level detection indicated a lower-only tone case.
- `visual_segment_selected_as_lower`: the row had upper+lower tone evidence, and
  the syllable segment had a lower-position tone component.
- `visual_segment_single_tone__lower_null`: the row had upper+lower tone
  evidence somewhere, but this syllable segment did not show a lower component,
  so the selected tone is put in `upper_tone`.
- `tone_position_unknown__lower_null`: no crop image was available, so the
  selected tone is kept as upper by default and the row remains review-only.

## Why This Helps OCR

The old flat label `ipa_digits` mixes two tasks:

1. Recognize the IPA letters and selected tone digits.
2. Decide whether the selected digit comes from the upper/right or lower/right
   printed position.

The structured schema separates those concerns. A model can still output the
final selected Wu-pinyin-compatible value, while the data keeps explicit slot
information for future physical OCR.

## Detector Route

Detecting whether a syllable has a lower/right tone number should be relatively
easy compared with full OCR:

- Binarize the crop or use Otsu thresholding.
- Estimate the main IPA baseline from tall connected components.
- Extract small digit-like connected components.
- Components above the baseline are upper/right tone candidates.
- Components below the baseline are lower/right tone candidates.
- Split the crop into syllable spans left-to-right and mark each syllable as
  `has_lower_tone` when a lower component falls in its span.

The current script already implements a heuristic version of this in
`ipa_ocr_work/scripts/build_structured_tone_labels.py`. It is good enough to
generate reviewable labels, but it should not yet be treated as perfect ground
truth.

## Model Strategy

Recommended near-term route:

1. Train a main OCR model to output `ipa_base + selected_tone`, using the manual
   Wu-pinyin-derived label as truth.
2. Train or refine a lightweight tone-position detector that predicts
   `has_lower_tone` per syllable.
3. Combine them into structured output:
   - if `has_lower_tone=true`, put the selected tone in `lower_tone`;
   - otherwise put it in `upper_tone` and set `lower_tone=null`.

This two-stage route is likely to perform better than asking one sequence model
to infer everything at once, because the lower/right decision is a geometric
layout problem, while IPA recognition is a character recognition problem. The
detector can use simple visual evidence and requires far less data.

Alternative route:

- A single multi-task model with a shared image encoder and two heads:
  one CTC head for `ipa_base + selected_tone`, and one binary/sequence head for
  per-syllable lower-tone presence.

This may be best later, but it needs cleaner syllable boundaries and more
careful evaluation. For the current data quality, a detector plus OCR pipeline
is safer and easier to debug.

## Caveat

The manual Wu-pinyin column is the source of truth for `selected_tone`. The unselected physical tone, especially the upper tone in upper+lower cases, is not yet labeled and remains null. A future physical-tone OCR pass should fill that slot if full visual transcription is needed.

## Generated Files

- `ipa_ocr_work/dataset/shaoxing_structured_tone_labels/structured_tone_labels.jsonl`
- `ipa_ocr_work/dataset/shaoxing_structured_tone_labels/structured_tone_syllables.tsv`
- `ipa_ocr_work/dataset/shaoxing_structured_tone_labels/summary.txt`
