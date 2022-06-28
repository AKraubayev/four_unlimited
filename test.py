import urllib.request
import zipfile
import os
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


url = 'https://www.dropbox.com/s/ni9567tj2x2r5b6/ml_engineering_weapon_and_no.zip'
urllib.request.urlretrieve(url, 'ml_engineering_weapon_and_no.zip')


os.mkdir('Models/', mode=777)
with zipfile.ZipFile('ml_engineering_weapon_and_no.zip', 'w') as zip_ref:
    zip_ref.extractall('Models/ml_engineering_weapon_and_no')


def load_trained_model():
    model = load_model("Models/ml_engineering_weapon_and_no")
    return model


model = load_trained_model()


def test_civilian_image():
    # Загрузка и преобразование изображения не оружия
    img_path = 'train/noweapon/File 1710.jpg'
    img = image.load_img(img_path)
    x = preprocess_image(img)
    pred = model.predict(x)
    assert pred < 0.2


def test_weapon_image():
    # Загрузка и преобразование изображения оружия
    img_path = 'train/weapon/File 1016.jpg'
    img = image.load_img(img_path)
    x = preprocess_image(img)
    pred = model.predict(x)
    assert pred > 0.9
