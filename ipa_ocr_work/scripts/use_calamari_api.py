import sys
from calamari_ocr.ocr.predict import Predictor

# 使用绝对路径
model_path = "e:\\my_pro\\ocr-ipa\\model\\calamari\\best.ckpt"
img_path = "e:\\my_pro\\moon_correct.png"

print(f"Loading model from {model_path}...")
predictor = Predictor.from_paths([model_path])

print(f"Processing image from {img_path}...")
predictions = predictor.predict([img_path])

for i, pred in enumerate(predictions):
    print(f"Prediction {i}:")
    print(f"  Result: {pred.sentence}")
    print(f"  Confidence: {pred.avg_char_probability}")
