# Shaoxing OCR Path To Usable Accuracy

## Current Evidence

The best fair test result is the TrOCR IPA continuation:

- run: `E3_trocr_base_printed_ipa_pad_square_continue_best_b4_lr3e6_e10`
- test Wu-pinyin row exact: `0.4414`
- test Wu-pinyin CER: `0.2661`
- test IPA row exact: `0.4414`
- test IPA CER: `0.2390`

This proves the labels and scoring path are usable, but also shows that
pad-square whole-row TrOCR is not a sufficient mainline. Long rows lose too
much horizontal detail after being squeezed into a square image, and free-form
generation compounds small syllable errors into row failures.

## Practical Target

For production usefulness, the target should be:

- final Wu-pinyin row exact `>= 0.80`
- Wu-pinyin CER `<= 0.08`
- tone-position detector fixed at geometric detector accuracy `1.0`
- evaluation always on the unchanged test split

## Main Route

The highest-confidence route is not another small TrOCR LR search. It is:

1. Build reliable syllable crops.
2. Recognize each syllable as a closed-set class.
3. Concatenate syllable predictions back into rows.
4. Apply a legal Wu-pinyin syllable decoder.
5. Use row-level OCR only as fallback or ensemble signal.

Why this is likely to move accuracy:

- The legal syllable inventory is finite.
- The tone-position detector has already reached perfect measured accuracy.
- Short crops avoid the 384-square long-row compression problem.
- Closed-set decoding prevents invalid IPA/Wu-pinyin hallucinations.
- Row exact improves multiplicatively when each syllable error rate drops.

The rough math is unforgiving but useful. If average rows have 2-3 syllables,
row exact `0.80` requires syllable exact around `0.90-0.93`. That is a clear
training target for the classifier and a better objective than hoping a
free-form row generator learns all constraints.

## Backup Route

The second route is a variable-width CTC recognizer:

- keep aspect ratio
- normalize height to 64
- allow width up to 960 or higher
- train `svtr_tiny` or CRNN/ResNet-Transformer CTC
- score with `score_ocr_experiment.py`
- apply `apply_wupin_lexicon_decoder.py`

This directly addresses the same compression failure as TrOCR, but still has a
harder long-sequence recognition problem than syllable classification.

## TrOCR Role

TrOCR should stay as a control and possible ensemble input, not as the main
path. The current best `0.4414` is useful as a baseline. Continue it only if
the new server is idle or if testing a specific preprocessing hypothesis.

## New Scripts

`ipa_ocr_work/scripts/run_usable_ocr_experiments.ps1`

Runs the decision tree on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File ipa_ocr_work\scripts\run_usable_ocr_experiments.ps1 `
  -Root G:\shaoxing_tts `
  -RunName usable_ocr_path `
  -MaxHours 8 `
  -Batch 192
```

Smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File ipa_ocr_work\scripts\run_usable_ocr_experiments.ps1 `
  -Root G:\shaoxing_tts `
  -RunName smoke_usable_ocr_path `
  -SmokeOnly
```

`ipa_ocr_work/scripts/run_usable_ocr_experiments.sh`

Runs the same decision tree on Linux/WSL:

```bash
bash ipa_ocr_work/scripts/run_usable_ocr_experiments.sh \
  --root /path/to/shaoxing_tts \
  --run-name usable_ocr_path \
  --max-hours 8 \
  --batch 192
```

Linux smoke test:

```bash
bash ipa_ocr_work/scripts/run_usable_ocr_experiments.sh \
  --root /path/to/shaoxing_tts \
  --run-name smoke_usable_ocr_path \
  --smoke-only \
  --skip-trocr \
  --skip-row-ctc
```

`ipa_ocr_work/scripts/apply_wupin_lexicon_decoder.py`

Converts model predictions into legal Wu-pinyin using a train-split syllable
lexicon and optional row-nearest correction:

```powershell
python ipa_ocr_work\scripts\apply_wupin_lexicon_decoder.py `
  --eval-manifest ipa_ocr_work\dataset\shaoxing_dual_model\ocr_selected\eval_manifest.tsv `
  --predictions ipa_ocr_work\runs\RUN\E2_variable_width_ctc_svtr_tiny\predictions_original_export.tsv `
  --out ipa_ocr_work\runs\RUN\E2_variable_width_ctc_svtr_tiny\predictions_lexicon.tsv `
  --prediction-mode ipa `
  --lexicon-split train `
  --row-nearest
```

`ipa_ocr_work/scripts/package_usable_ocr_experiments.ps1`

Builds a transfer zip for a new server:

```powershell
powershell -ExecutionPolicy Bypass -File ipa_ocr_work\scripts\package_usable_ocr_experiments.ps1 `
  -Root G:\shaoxing_tts
```

## Decision Tree

1. Run smoke test on the new server.
2. Run `E1_syllable_closed_set`.
3. If syllable test exact reaches `>=0.90`, make syllable pipeline the mainline.
4. If syllable exact is below `0.85`, inspect crop quality before training more.
5. Run `E2_variable_width_ctc_svtr_tiny`.
6. If E2 row exact with lexicon decoder beats TrOCR by `>=0.05`, continue CTC.
7. If both E1 and E2 stall, the bottleneck is crop/segmentation quality, not OCR
   backbone choice.

## Expected Next Improvements

Most likely gains:

- better clustered syllable crops
- classifier augmentation: blur, threshold jitter, small crop jitter
- classifier pretrained backbone if available on the new server
- CTC variable-width ensemble with syllable classifier
- legal syllable decoder and train-split lexicon correction

This route is the most credible path from `0.44` to a usable `0.80+`, because
it changes the problem from unconstrained whole-row generation into constrained
short-unit recognition.
