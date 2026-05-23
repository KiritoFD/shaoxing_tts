import pandas as pd
import re

print("=" * 60)
print("将Calamari OCR的IPA识别结果合并到Excel")
print("=" * 60)

# 1. 读取PaddleOCR结果（用于获取汉字词条）
paddle_results = {}
with open('e:/my_pro/ipa_ocr_work/images/ipa_candidates_gpu2/ocr_results.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 3:
            idx = int(parts[0])
            text = parts[1]
            paddle_results[idx] = text

print(f"读取到 {len(paddle_results)} 个PaddleOCR结果")

# 2. 读取Calamari OCR的IPA识别结果
ipa_results = {}
with open('e:/my_pro/ipa_ocr_work/results/batch_recognition_results.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    blocks = content.split('------------------------------------------------------------')
    for block in blocks:
        if '文件:' in block and '过滤结果:' in block:
            file_match = re.search(r'文件:\s*(\S+)', block)
            filtered_match = re.search(r'过滤结果:\s*(\S*)', block)
            if file_match and filtered_match:
                filename = file_match.group(1)
                filtered_ipa = filtered_match.group(1).strip()
                ipa_results[filename] = filtered_ipa

print(f"读取到 {len(ipa_results)} 个Calamari OCR结果")

# 3. 建立索引到IPA的映射
idx_to_ipa = {}
for filename, ipa in ipa_results.items():
    match = re.search(r'ipa_candidate_(\d+)\.png', filename)
    if match:
        idx = int(match.group(1))
        idx_to_ipa[idx] = ipa

print(f"解析到 {len(idx_to_ipa)} 个索引-IPA映射")

# 4. 建立汉字到IPA的映射
hanzi_to_ipa = {}
for idx, text in paddle_results.items():
    if idx in idx_to_ipa:
        hanzi = ''
        for c in text:
            if '\u4e00' <= c <= '\u9fff':
                hanzi += c
            else:
                break
        if hanzi:
            hanzi_to_ipa[hanzi] = idx_to_ipa[idx]

print(f"\n汉字-IPA映射 ({len(hanzi_to_ipa)} 个):")
for hanzi, ipa in hanzi_to_ipa.items():
    print(f"  {hanzi} -> {ipa}")

# 5. 读取Excel
df = pd.read_excel('e:/my_pro/result_all_converted.xlsx')
df_with_page = df[df['页码'].notna()].copy()

# 6. 获取第124页的词条
target_page = 127
page_123 = df_with_page[df_with_page['页码'] == target_page].copy()
print(f"\n第{target_page}页有 {len(page_123)} 个词条")

# 7. 匹配到Excel
matched_count = 0
for idx, row in page_123.iterrows():
    hanzi = str(row['汉字']).strip()
    if hanzi in hanzi_to_ipa:
        df_with_page.loc[idx, 'IPA识别'] = hanzi_to_ipa[hanzi]
        matched_count += 1

print(f"成功匹配 {matched_count} 个词条")

# 8. 保存结果
df_with_page.to_excel('e:/my_pro/result_all_converted.xlsx', index=False)
print("结果已保存到 e:/my_pro/result_all_converted.xlsx")

# 9. 显示匹配结果
print("\n" + "=" * 60)
print("匹配结果预览:")
print("=" * 60)
result_df = df_with_page[df_with_page['页码'] == target_page]
result_df = result_df[result_df['IPA识别'].notna()]
print(result_df[['汉字', 'IPA识别']].to_string(index=False))
