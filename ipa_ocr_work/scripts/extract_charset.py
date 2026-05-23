import json

with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r') as f:
    data = json.load(f)

print('data keys:', list(data['scenario']['data'].keys())[:20])
if 'codec' in data['scenario']['data']:
    print('codec found in data')
    if 'charset' in data['scenario']['data']['codec']:
        charset = data['scenario']['data']['codec']['charset']
        print(f'Charset length: {len(charset)}')
        print('Index 96:', repr(charset[96]))
        print('Index 99:', repr(charset[99]))
        print('Index 387:', repr(charset[387]))
        with open('e:/my_pro/charset.txt', 'w', encoding='utf-8') as f:
            f.write(''.join(charset))
        print('Saved charset to charset.txt')
