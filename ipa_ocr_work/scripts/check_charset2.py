import json

with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r') as f:
    data = json.load(f)

charset = data['scenario']['data']['codec']['charset']
print(f'Charset length: {len(charset)}')
print(f'First 100 characters (indices 0-99):')
for i in range(min(100, len(charset))):
    print(f'  [{i}] {repr(charset[i])}')

# Let's also check if there are any standard Latin letters
print(f'\nChecking for Latin letters:')
letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
found_letters = []
for i, c in enumerate(charset):
    if c in letters:
        found_letters.append((i, c))
print(f'Found {len(found_letters)} Latin letters:')
for i, c in found_letters[:20]:
    print(f'  [{i}] {c}')
