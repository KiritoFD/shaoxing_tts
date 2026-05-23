import json

with open('e:/my_pro/ocr-ipa/model/calamari/best.ckpt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

charset = data['scenario']['data']['codec']['charset']

with open('e:/my_pro/charset_list.txt', 'w', encoding='utf-8') as out:
    out.write(f"字符集大小: {len(charset)} 个字符\n\n")
    out.write("="*60 + "\n")
    out.write("完整字符集列表:\n")
    out.write("="*60 + "\n")

    for i, char in enumerate(charset):
        char_repr = repr(char)
        if len(char) == 1:
            ord_val = ord(char)
            if ord_val < 128:
                char_type = "ASCII"
            elif '\u0250' <= char <= '\u02ee':
                char_type = "IPA字母"
            elif '\u0300' <= char <= '\u036f':
                char_type = "IPA组合符"
            elif '\u00c0' <= char <= '\u00ff':
                char_type = "拉丁补充"
            elif char in 'ⁿ⁾':
                char_type = "上标"
            elif char == '◌':
                char_type = "组合省略号"
            elif char == '∅':
                char_type = "空集"
            elif char == 'ꜜ':
                char_type = "音标"
            elif ord_val > 10000:
                char_type = "特殊符号"
            else:
                char_type = "其他"
        else:
            ord_val = 0
            char_type = "组合字符"

        out.write(f"索引 {i:3d}: {char_repr:10s} (U+{ord_val:05X}) - {char_type}\n")

    out.write("\n" + "="*60 + "\n")
    out.write("按类型统计:\n")
    out.write("="*60 + "\n")

    ipa_letters = []
    ipa_modifiers = []
    ascii_chars = []
    other_chars = []

    for i, char in enumerate(charset):
        if len(char) == 1:
            ord_val = ord(char)
            if ord_val < 128:
                ascii_chars.append((i, char))
            elif '\u0250' <= char <= '\u02ee':
                ipa_letters.append((i, char))
            elif '\u0300' <= char <= '\u036f':
                ipa_modifiers.append((i, char))
            else:
                other_chars.append((i, char))
        else:
            other_chars.append((i, char))

    out.write(f"ASCII字符: {len(ascii_chars)} 个\n")
    out.write(f"IPA字母: {len(ipa_letters)} 个\n")
    out.write(f"IPA修饰符: {len(ipa_modifiers)} 个\n")
    out.write(f"其他字符: {len(other_chars)} 个\n")

print("字符集已保存到 e:/my_pro/charset_list.txt")
