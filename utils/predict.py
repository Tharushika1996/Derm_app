import cv2
import numpy as np
import tensorflow as tf

labels = ['1. Eczema 1677',
          '10. Warts Molluscum and other Viral Infections - 2103',
          '2. Melanoma 15.75k',
          '3. Atopic Dermatitis - 1.25k',
          '4. Basal Cell Carcinoma (BCC) 3323',
          '5. Melanocytic Nevi (NV) - 7970',
          '6. Benign Keratosis-like Lesions (BKL) 2624',
          '7. Psoriasis pictures Lichen Planus and related diseases - 2k',
          '8. Seborrheic Keratoses and other Benign Tumors - 1.8k',
          '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k']


def predict(image_path):
    # read image
    image = cv2.imread(image_path)
    image = np.array(image)

    # reshape to 128x128
    image = cv2.resize(image, (128, 128))

    # scale
    image = image / 255.0

    # load model
    model = tf.keras.models.load_model("model.h5", compile=False)

    # predict
    prediction = model.predict(np.array([image]))
    # argmax
    prediction = np.argmax(prediction)
    # label
    label = labels[prediction]
    # prediction percentage
    prediction_percentage = np.max(model.predict(np.array([image])))

    return label, prediction_percentage

