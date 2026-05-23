import json

with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r') as f:
    data = json.load(f)
charset = data['scenario']['data']['codec']['charset']

print('Checking specific characters in charset:')
indices = [0, 54, 62, 67, 71, 73, 387]
for idx in indices:
    if idx < len(charset):
        c = charset[idx]
        if len(c) > 0:
            print(f'Index {idx}: {repr(c)} (U+{ord(c):04X})')
        else:
            print(f'Index {idx}: EMPTY STRING')
    else:
        print(f'Index {idx}: OUT OF RANGE (charset length: {len(charset)})')

print(f'\nTotal charset length: {len(charset)}')
print(f'Last 10 characters:')
for i in range(max(0, len(charset)-10), len(charset)):
    c = charset[i]
    if len(c) > 0:
        print(f'  [{i}]: {repr(c)} (U+{ord(c):04X})')
    else:
        print(f'  [{i}]: EMPTY STRING')
