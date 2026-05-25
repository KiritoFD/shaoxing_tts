# PaddleOCR fine-tuning runs

## 2026-05-24 ppocrv5_mobile_ipa_b192_e80_20260524_1549

- Host: `connect.bjb2.seetacloud.com:35383`, RTX 5090.
- Path: `/root/autodl-tmp/shaoxing_tts/paddlex_runs/ppocrv5_mobile_ipa_b192_e80_20260524_1549`
- Dataset: `ocr_selected_all`, mixed quality, includes `low_match`.
- Split counts: train 3288, val 224, test 261.
- Model: PP-OCRv5 mobile recognition pretrained fine-tune.
- Batch: 192. Batch 256 previously OOMed on the 32 GB card.
- Status: no active training process; GPU idle at check time.
- Best Paddle validation metric before stop:
  - epoch: 38
  - acc: 0.30357141501913326
  - norm_edit_dis: 0.7706690574948258
  - fps: 1100.2174568824184
- Last observed training log: epoch 40, global_step 950, max_mem_reserved 26009 MB, max_mem_allocated 24155 MB.
- Caveat: this is Paddle internal validation on the old mixed-quality dataset, not final fixed-test Wu-pinyin row accuracy.

Next run should switch to `ocr_selected_phonetic_reliable`, which drops `low_match` and ignores Chinese headword mismatch because the OCR target is the phonetic crop.

## 2026-05-24 ppocrv5_mobile_ipa_phonetic_reliable_b192_e80_20260524_160415

- Host: `connect.bjb2.seetacloud.com:35383`, RTX 5090.
- Path: `/root/autodl-tmp/shaoxing_tts/paddlex_runs/ppocrv5_mobile_ipa_phonetic_reliable_b192_e80_20260524_160415`
- Dataset: `ocr_selected_phonetic_reliable`; keeps `matched` + `weak_match`, drops `low_match`.
- Split counts: train 2590, val 213, test 251.
- Model: PP-OCRv5 mobile recognition pretrained fine-tune.
- Batch: 192.
- Status: completed 80 epochs; GPU idle after completion.
- Best Paddle validation metric:
  - epoch: 76
  - acc: 0.38967134320791813
  - norm_edit_dis: 0.7963484223398773
- Fixed test scoring from `best_accuracy` checkpoint:
  - IPA row exact: 0.3147410358565737
  - IPA CER: 0.25101214574898784
  - Wu-pinyin row exact: 0.3147410358565737
  - Wu-pinyin CER: 0.30146609651802075
- Caveat: this improves over the old mixed-quality Paddle validation metric, but is still below the prior best TrOCR fixed-test row exact and far below usable OCR quality.
