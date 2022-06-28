# -*- coding: utf-8 -*-
""" "ml_engineering_weapon_and_no.ipynb""

Original file is located at
    https://colab.research.google.com/drive/145c7vHHGECzQp_X4vGuCBD0UIS_XmlIH
## Задание с хакатона осеннего семестра первого курса магистратуры "Инженерия машинного обучения":
# No/Weapons: набор изображений, содержащих и не содержащих оружие #
## Версия: 2021.11.22.0 ##
Наборы изображений с оружием и не с оружием. В наборы содержащих оружие (папки "weapon") в датасетах train и test,включены изображение следующих предмметов: Пистолет, Пистолет-пулемёт, Автомат, Автоматическая винтовка, РПГ, ПЗРК,Винтовка, Ружьё, Пулемёт, Гранатомёт, Кинжал, Тесак, Мачете, Граната, Метательный нож.В наборы НЕ содержащих оружие (папки "noweapon") в соответствующих датасетах,включены в том числе,следующие иображения: Сигарета, Ручка,Ручка в руке,Стилус,Шприц,Ножницы,Стилус в руке, Фотоапарат, Дерево-бонсай, Мозг, Люстра, Бабочка и др.
## Свойства набора данных ## 
Общее количество изображений: 2675.
Размер обучающего набора: 2280 изображения (допустимо несколько изображений предмета), из них 964 изображения пренадлежат к классу "не оружие" и расположены в папке test\noweapon и 1316 , к классу "оружие" и расположены в папке test\weapon
Размер тестового набора: 365 изображений (допустимо несколько изображений предмета), из них 95 изображения пренадлежат к классу "не оружие", расположены в папке test\noweapon и 300, к классу "оружие",расположены в папке test\weapon
Количество классов: 2 (оружие и не оружие, в коде: "weapon" и "noweapon" , соответственно).
## Структура репозитория ##
Архивы [Обучение] (train.zip) и [Тест] (test.zip) содержат все изображения, используемые для обучения и тестирования.В каждом из архивов изображения,преимущественно содержащие классифицируемый объект крупным планом,с очищенным или расфокусированным фоном,рассортированы по папкам "weapon" и "noweapon" в соответствии с классом.
## Подключаем библиотеки:
"""
import io
import urllib.request
import zipfile
import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions



"""## Конструируем функцию предобработки"""

def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


trg_s = (224, 224)
img_s = (224, 224)
batch_s = 128



"""## Загружаем обученную на датасете модель"""


url = 'https://www.dropbox.com/s/ni9567tj2x2r5b6/ml_engineering_weapon_and_no.zip'


os.mkdir('Models/', mode=777)
urllib.request.urlretrieve(url, 'ml_engineering_weapon_and_no.zip')
with zipfile.ZipFile('ml_engineering_weapon_and_no.zip', 'w') as zip_ref:
    zip_ref.extractall('Models/ml_engineering_weapon_and_no')
    

def load_trained_model():
    model = load_model("Models/ml_engineering_weapon_and_no")
    return model


"""# Использование нейронной сети для распознавания изображений"""


model = load_trained_model()


"""##Загружаем изображение из файла в StreamLit"""


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


"""##Запускаем предобработку и распознавание"""


st.title('**Классификация оружия на изображении**')
img = load_image()
result = st.button('Распознать изображение')
def print_percent(t):
    return {
               t >= 0.5: str(round(t, 4) * 100),
               t < 0.5: str(round(1 - t, 4) * 100),
           }[1] + "%"


"""Печатаем результаты распознавания"""


if result:
    x = preprocess_image(img)
    prediction = model.predict(x)
    x = prediction[0][0]
    sub = {
        x > 0.5 : "Это оружие",
        x == 0.5: "Не определено",
        x  < 0.5 : "Это НЕ оружие"
          }[1]
    st.write('**Результаты распознавания: \n **',sub + ", с вероятностью: " + print_percent(x))
