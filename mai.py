import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('cat_dog_model.h5')

# Load and preprocess the test image
img_path = r'C:\Users\yashw\Documents\bharatintern\New folder\test\dogs\dog.1301.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Use the model to make a prediction
prediction = model.predict(img_array)
if prediction[0][0] < 0.5:
    print('Cat')
else:
    print('Dog')