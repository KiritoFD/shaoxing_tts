import pandas as pd
import re

print("=" * 60)
print("根据识别结果内容匹配Excel词条")
print("=" * 60)

# 1. 读取批量识别结果
results_file = 'e:/my_pro/ipa_ocr_work/results/batch_recognition_results.txt'
with open(results_file, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'文件: (ipa_candidate_(\d+)_orig(\d+)\.png)\n原始识别: ([^\n]+)\n过滤结果: ([^\n]+)'
matches = re.findall(pattern, content)

print(f"找到 {len(matches)} 个识别结果")

# 2. 读取Excel
df = pd.read_excel('e:/my_pro/result_all_converted.xlsx')
df_with_page = df[df['页码'].notna()].copy()

# 添加IPA列
if 'IPA识别' not in df_with_page.columns:
    df_with_page['IPA识别'] = ''

# 3. 获取第123页的词条
page_123 = df_with_page[df_with_page['页码'] == 123]
page_123_indices = page_123.index.tolist()
print(f"第123页有 {len(page_123)} 个词条")

# 4. 根据识别结果内容推断对应的词条
# 从识别结果看，包含ɦiøʔliaŋ的应该是"月亮"相关的词条

# 建立关键词映射
keyword_map = {
    'ɦiøʔliaŋbobo': '月亮婆婆',
    'ɦiøʔliaŋkuɒŋ': '月亮光',
    'ɦiøʔliaŋldiɦo': '月亮地下',
    'ɦiøʔliaŋ': '月亮',
    'ɦiøʔŋo': '月华',
    'ɕiŋ': '星',
    'huɒŋhuẽɕiɒ': '黄昏晓',
    'huɒŋhuẽɕiŋ': '黄昏星',
    'foŋ': '风',
}

# 匹配识别结果到Excel词条
matched_count = 0
for filename, idx, orig_idx, raw, filtered in matches:
    idx = int(idx)
    orig_idx = int(orig_idx)
    
    # 根据过滤结果中的关键词匹配
    matched_hanzi = None
    for keyword, hanzi in keyword_map.items():
        if keyword in filtered:
            matched_hanzi = hanzi
            break
    
    if matched_hanzi:
        # 在Excel中查找对应的词条
        for excel_idx in page_123_indices:
            if df_with_page.loc[excel_idx, '汉字'] == matched_hanzi:
                df_with_page.loc[excel_idx, 'IPA识别'] = filtered
                print(f"匹配: {matched_hanzi} -> {filtered}")
                matched_count += 1
                break

print(f"\n成功匹配 {matched_count} 个词条")

# 5. 保存结果
df_with_page.to_excel('e:/my_pro/result_all_converted.xlsx', index=False)
print(f"\n结果已保存到 e:/my_pro/result_all_converted.xlsx")
