import tensorflow as tf
print('TF version:', tf.__version__)
print('SavedModel path: e:/my_pro/ocr-ipa/model/calamari/best.ckpt')
try:
    model = tf.saved_model.load('e:/my_pro/ocr-ipa/model/calamari/best.ckpt')
    print('Model type:', type(model))
    print('Signatures:', list(model.signatures.keys()) if hasattr(model, 'signatures') else 'No signatures')
except Exception as e:
    print('Error:', e)
