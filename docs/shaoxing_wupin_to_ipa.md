# Shaoxing Wu-Pinyin to IPA Conversion

This project treats the `wupin` column exported from `result_all_converted.xlsx`
as the trusted manual label. The legacy `IPA识别` column is not used as ground
truth.

## Sources

- Wu Chinese Association Shaoxing romanization:
  `https://wu-chinese.com/romanization/shaoxing.html`
- Wu romanization comparison table supplied by the user:
  `https://simonsunchn.github.io/posts/2026-01-29-%E5%90%B4%E8%AF%AD%E6%8B%BC%E9%9F%B3%E6%96%B9%E6%A1%88%E5%AF%B9%E7%85%A7%E8%A1%A8/post/`
- Annotator constraint: the OCR target IPA inventory is limited to
  `p pʰ b m f v t tʰ d n l s sʰ z ɕ ɕʰ ʑ ȵ k kʰ ɡ ŋ h ɦ ɿ a ᴇ ɤ ɒ o æ̃ ẽ ø̃ ə ø ʔ i u y`,
  plus tone digits or superscript tone digits.

## Important Assumptions

- The dataset is Shaoxing dialect, so the mapping follows the Shaoxing inventory,
  not a generic Suzhou/Shanghai Wu table.
- The manual labels use `q` for checked finals where the published Shaoxing page
  writes `h`, for example `oeq` corresponds to `oeh`.
- Tone digits are copied from the manual label. The user rule is: use the lower
  right number when present; otherwise use the upper right number. The converter
  does not reinterpret tones, it only preserves the digits as IPA superscripts.
- Some labels contain spelling variants or typos. These are handled explicitly
  only when they occur in the trusted data.
- Visual checks against the PDF showed several Shaoxing-specific whole-syllable
  spellings that cannot be recovered by naive initial/final splitting:
  `yu -> ɦy`, `yoeq -> ɦiøʔ`, `yoen -> ɦiø̃`, and the attested `dan -> dɒŋ`
  in words written 荡/宕.

## Current Core Rules

Initials:

| Wu-pinyin | IPA |
|---|---|
| p ph b m f v | p pʰ b m f v |
| t th d n l | t tʰ d n l |
| ts tsh dz s z | ts tsʰ dz s z |
| c ch j sh zh gn/ny | tɕ tɕʰ dʑ ɕ ʑ ȵ |
| k kh g ng | k kʰ ɡ ŋ |
| h gh w/y | h ɦ ɦ |

Main finals:

| Wu-pinyin | IPA |
|---|---|
| y i u iu/yu | ɿ i u y |
| a ua ia | a ua ia |
| o uo io | ɒ uɒ iɒ |
| e ue ie | ᴇ uᴇ iᴇ |
| au iau | ɤ iɤ |
| eu ieu | ə iə |
| aen uaen iaen | æ̃ uæ̃ iæ̃ |
| en/een uen ien | ẽ/uẽ/iẽ |
| oen uoen ioen | ø̃ uø̃ iø̃ |
| an uan ian | aŋ uaŋ iaŋ |
| aon uaon iaon | ɒŋ uɒŋ iɒŋ |
| on uon ion | oŋ uoŋ ioŋ |
| eon in | əŋ iŋ |

Checked finals:

| Wu-pinyin | IPA |
|---|---|
| aq/ah uaq/uah iaq/iah | aʔ uaʔ iaʔ |
| aeq/aeh uaeq/uaeh | æʔ uæʔ |
| eq/eh ueq/ueh iq/ih | ᴇʔ uᴇʔ iʔ |
| eoq/eoh | əʔ |
| oq/oh uoq/uoh ioq/ioh | oʔ uoʔ ioʔ |
| oeq/oeh uoeq/uoeh ioeq/ioeh | øʔ uøʔ iøʔ |

## Generated Files

- Mapping table:
  `ipa_ocr_work/config/wupin_ipa_map.json`
- Converter:
  `ipa_ocr_work/scripts/wupin_ipa_convert.py`
- Generated CSV:
  `result_all_converted.with_ipa.csv`
- Unknown report:
  `ipa_ocr_work/reports/wupin_ipa_unknowns.tsv`
