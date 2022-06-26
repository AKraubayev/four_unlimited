import io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

model = load_model("/Models/ml_engineering_weapon_and_no/")

def test_civilian_image():
    # Загрузка и преобразование изображения не оружия
    img_path = '/train/noweapon/File 1710.jpg'
    img = image.load_img(img_path)
    x = preprocess_image(img)
    pred = model.predict(x)
    assert pred < 0.2


def test_weapon_image():
    # Загрузка и преобразование изображения оружия
    img_path = '/train/weapon/File 1016.jpg'
    img = image.load_img(img_path)
    x = preprocess_image(img)
    pred = model.predict(x)
    assert pred > 0.9
