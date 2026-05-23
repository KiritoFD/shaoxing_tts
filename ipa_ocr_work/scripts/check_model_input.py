import tensorflow as tf

model_path = 'e:/my_pro/ocr-ipa/model/calamari/best.ckpt'
print('Loading model...')
model = tf.saved_model.load(model_path)

print('\nModel signatures:')
for key in model.signatures.keys():
    print(f'\nSignature: {key}')
    concrete_func = model.signatures[key]
    print('Inputs:')
    for input_tensor in concrete_func.inputs:
        print(f'  {input_tensor.name}: {input_tensor.shape}')
    print('Outputs:')
    for output_tensor in concrete_func.outputs:
        print(f'  {output_tensor.name}: {output_tensor.shape}')
