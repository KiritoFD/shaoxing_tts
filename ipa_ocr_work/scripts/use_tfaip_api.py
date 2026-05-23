import json
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
from tfaip import Scenario

# Load the model
model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'

print('Loading model...')
scenario = Scenario.from_model_path(model_path)
model = scenario.create_model()
print('Model loaded successfully')

# Load weights
model.load_weights(model_path)
print('Weights loaded successfully')

# Set up the data pipeline
data_pipeline = scenario.create_data_pipeline(predict=True)

# Load and preprocess image
img_path = 'e:/my_pro/moon_correct.png'
print(f'Loading image from {img_path}')

img = Image.open(img_path).convert('L')
img_np = np.array(img, dtype=np.uint8)

# Apply the preprocessing pipeline
prepared = data_pipeline.processors.process([img_np], meta=[{}])
print(f'Prepared data shape: {prepared[0].shape}')

# Make predictions
predictions = model.predict(prepared[0])
print('Predictions:', predictions)
