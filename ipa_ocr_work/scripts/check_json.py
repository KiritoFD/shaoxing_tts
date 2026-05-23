import json

with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r') as f:
    data = json.load(f)

print('Top level keys:', list(data.keys())[:20])

if 'scenario' in data:
    print('scenario keys:', list(data['scenario'].keys())[:20])
    if 'model' in data['scenario']:
        print('model keys:', list(data['scenario']['model'].keys())[:20])
