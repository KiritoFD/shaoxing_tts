from PIL import Image
import numpy as np

# Load the test image
img_path = 'e:/my_pro/test_generated.png'
img = Image.open(img_path).convert('L')
img_array = np.array(img)

print(f'Test image shape: {img_array.shape}')
print(f'Test image dtype: {img_array.dtype}')
print(f'Min value: {img_array.min()}, Max value: {img_array.max()}')

# Save a copy for visual inspection
img.save('e:/my_pro/test_generated_copy.png')

# Print some statistics
print(f'Non-white pixels: {np.sum(img_array < 250)}')
print(f'First 10 rows, first 20 cols:')
print(img_array[:10, :20])
