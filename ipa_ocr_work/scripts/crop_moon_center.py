from PIL import Image
import numpy as np

# Load the already cropped moon_correct.png
img_path = 'e:/my_pro/moon_correct.png'
img = Image.open(img_path).convert('L')
img_array = np.array(img)

print(f"Original moon_correct size: {img.size}")
print(f"Original shape: {img_array.shape}")

# moon_correct.png is already cropped to y=226-319 (2x) from the PDF
# But it's 2000px wide, which is too wide
# Let's crop it to a reasonable width

# Let's crop it to the middle portion
# The text should be roughly in the middle
width = img_array.shape[1]
center = width // 2
half_width = 300  # 300 pixels on each side

# Crop to center 600 pixels
cropped = img_array[:, center-half_width:center+half_width]
cropped_img = Image.fromarray(cropped)
cropped_img.save('e:/my_pro/moon_center_600.png')

print(f"Cropped to: {cropped.shape}")
