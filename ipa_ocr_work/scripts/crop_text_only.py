from PIL import Image
import numpy as np

# Load the already cropped moon_center_600.png
img_path = 'e:/my_pro/moon_center_600.png'
img = Image.open(img_path).convert('L')
img_array = np.array(img)

print(f"Original size: {img.size}")

# Find text columns
col_sums = np.sum(img_array < 250, axis=0)
text_cols = np.where(col_sums > 0)[0]

if len(text_cols) > 0:
    start_col = text_cols[0]
    end_col = text_cols[-1] + 1
    print(f"Text region: columns {start_col} to {end_col}")

    # Crop to text region
    cropped = img_array[:, start_col:end_col]
    cropped_img = Image.fromarray(cropped)
    cropped_img.save('e:/my_pro/moon_text_only.png')

    print(f"Cropped image size: {cropped_img.size}")
    print(f"Cropped shape: {cropped.shape}")

    # Check non-white pixels ratio
    non_white = np.sum(cropped < 250)
    total = cropped.shape[0] * cropped.shape[1]
    print(f"Non-white ratio: {non_white}/{total} = {non_white/total:.2%}")
