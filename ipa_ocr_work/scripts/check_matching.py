import pandas as pd
import re

print("=" * 60)
print("合并IPA识别结果到Excel（根据内容匹配）")
print("=" * 60)

# 1. 读取批量识别结果
results_file = 'e:/my_pro/ipa_ocr_work/results/batch_recognition_results.txt'
with open(results_file, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'文件: (ipa_candidate_\d+_orig\d+\.png)\n原始识别: ([^\n]+)\n过滤结果: ([^\n]+)'
matches = re.findall(pattern, content)

print(f"找到 {len(matches)} 个识别结果")

# 显示所有识别结果
print("\n所有识别结果:")
for i, (filename, raw, filtered) in enumerate(matches):
    print(f"{i}: {filtered[:50]}")

# 2. 读取Excel
df = pd.read_excel('e:/my_pro/result_all_converted.xlsx')
df_with_page = df[df['页码'].notna()].copy()

# 添加IPA列
if 'IPA识别' not in df_with_page.columns:
    df_with_page['IPA识别'] = ''

# 3. 获取第123页的词条
page_123 = df_with_page[df_with_page['页码'] == 123]
print(f"\n第123页有 {len(page_123)} 个词条:")
for i, (idx, row) in enumerate(page_123.iterrows()):
    print(f"{i}: {row['汉字']}")

# 4. 手动建立对应关系（根据识别结果的内容推断）
# 从识别结果看：
# - 文件1: lɦiøʔliaŋkȵiøʔliaŋ -> 包含ɦiøʔliaŋ，应该是"月亮"
# - 文件2: ɦiøʔliaŋboboŋiøʔliaŋbobol -> 包含ɦiøʔliaŋbobo，应该是"月亮婆婆"

# 让我检查一下识别结果文件中的原始识别内容
print("\n检查原始识别内容:")
for i, (filename, raw, filtered) in enumerate(matches[:5]):
    print(f"{i}: {raw[:100]}")
