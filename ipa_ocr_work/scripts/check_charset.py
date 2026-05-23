import json

with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

charset = data['scenario']['data']['codec']['charset']
print(f"Charset size: {len(charset)}")
print(f"Index 387: {repr(charset[387])}")
print(f"Index 387 ord: {ord(charset[387]) if len(charset[387]) == 1 else 'not a single char'}")

# Print chars around index 387
print("\nChars around index 387:")
for i in range(380, 395):
    if i < len(charset):
        print(f"  Index {i}: {repr(charset[i])}")
