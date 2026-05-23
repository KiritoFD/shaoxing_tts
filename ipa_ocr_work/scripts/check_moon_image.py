from PIL import Image
import numpy as np

# Load the moon image
img_path = 'e:/my_pro/moon_correct.png'
img = Image.open(img_path).convert('L')
img_array = np.array(img)

print(f'Moon image shape: {img_array.shape}')
print(f'Moon image dtype: {img_array.dtype}')
print(f'Min value: {img_array.min()}, Max value: {img_array.max()}')

# Save a copy for visual inspection
img.save('e:/my_pro/moon_copy.png')

# Print some statistics
print(f'Non-white pixels: {np.sum(img_array < 250)}')
print(f'Image dimensions: {img_array.shape[1]}x{img_array.shape[0]} (width x height)')
