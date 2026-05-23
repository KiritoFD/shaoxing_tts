from PIL import Image
import numpy as np

# Load the cropped images for comparison
img1 = Image.open('e:/my_pro/moon_correct.png').convert('L')
img2 = Image.open('e:/my_pro/moon_center_600.png').convert('L')

arr1 = np.array(img1)
arr2 = np.array(img2)

print(f"moon_correct.png: {img1.size}, shape {arr1.shape}")
print(f"moon_center_600.png: {img2.size}, shape {arr2.shape}")

# Check pixel distribution
# Non-white pixels (text) in each
non_white1 = np.sum(arr1 < 250)
non_white2 = np.sum(arr2 < 250)

print(f"\nNon-white pixels in moon_correct.png: {non_white1}")
print(f"Non-white pixels in moon_center_600.png: {non_white2}")

# Save both for visual inspection
img1.save('e:/my_pro/check_moon_correct.png')
img2.save('e:/my_pro/check_moon_center_600.png')

# Try to find the text region by column
print(f"\nAnalyzing column-wise text distribution in moon_center_600.png...")
col_sums = np.sum(arr2 < 250, axis=0)
text_cols = np.where(col_sums > 0)[0]
if len(text_cols) > 0:
    print(f"Text found in columns {text_cols[0]} to {text_cols[-1]}")
    print(f"Total columns with text: {len(text_cols)} out of {len(col_sums)}")
else:
    print("No text found in image!")
