import pandas as pd
import re

print("=" * 60)
print("合并IPA识别结果到Excel")
print("=" * 60)

# 1. 读取批量识别结果
results_file = 'e:/my_pro/ipa_ocr_work/results/batch_recognition_results.txt'
with open(results_file, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'文件: (ipa_candidate_\d+_orig\d+\.png)\n原始识别: ([^\n]+)\n过滤结果: ([^\n]+)'
matches = re.findall(pattern, content)

print(f"找到 {len(matches)} 个识别结果")

# 2. 读取Excel
df = pd.read_excel('e:/my_pro/result_all_converted.xlsx')
df_with_page = df[df['页码'].notna()].copy()
print(f"有页码的记录数: {len(df_with_page)}")

# 添加IPA列
if 'IPA识别' not in df_with_page.columns:
    df_with_page['IPA识别'] = ''

# 3. 获取第123页的词条索引
page_123_indices = df_with_page[df_with_page['页码'] == 123].index.tolist()
print(f"第123页有 {len(page_123_indices)} 个词条")

# 4. 匹配识别结果到Excel
# 假设识别结果的顺序和Excel词条的顺序一致
matched_count = 0
for i, (filename, raw, filtered) in enumerate(matches):
    if i < len(page_123_indices):
        excel_idx = page_123_indices[i]
        hanzi = df_with_page.loc[excel_idx, '汉字']
        df_with_page.loc[excel_idx, 'IPA识别'] = filtered
        print(f"匹配: {hanzi} -> {filtered}")
        matched_count += 1

print(f"\n成功匹配 {matched_count} 个词条")

# 5. 保存结果
df_with_page.to_excel('e:/my_pro/result_all_converted.xlsx', index=False)
print(f"\n结果已保存到 e:/my_pro/result_all_converted.xlsx")
