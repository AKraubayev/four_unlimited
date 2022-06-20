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

# Commented out IPython magic to ensure Python compatibility.
import io
#import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential, load_model, 
import matplotlib.pyplot as plt

"""## Загружаем датасеты """

unzip -q train.zip
unzip -q test.zip

"""## Конструием функцию предобработки"""

def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


trg_s = (224, 224)
img_s = (224, 224)
batch_s = 128


"""## Смотрим примеры картинок:"""

img_path = '/train/weapon/File 1016.jpg'
img = image.load_img(img_path, target_size=trg_s)
plt.imshow(img)
plt.show()

"""## Создаем наборы данных
Набор данных для обучения
"""

train_dataset = image_dataset_from_directory('train',
                                             batch_size=batch_s,
                                             image_size=img_s)

class_names = train_dataset.class_names

class_names

"""Набор данных для тестирования"""

test_dataset = image_dataset_from_directory('test',
                                             batch_size=batch_s,
                                             image_size=img_s)

"""## Создаем составную нейронную сеть"""

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

inputs = layers.Input(shape=(224, 224, 3))
x = img_augmentation(inputs)
model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

# Freeze the pretrained weights
model.trainable = False

# Rebuild top
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization()(x)

top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
# Для задачи с двумя классами изображений
outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
# Для задачи с несколькими классами изображений
# num_classes = 3 # Задаем количество классов
# outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
model = tf.keras.Model(inputs, outputs, name="EfficientNet")

"""Компилируем составную нейронную сеть"""

# Для задачи с двумя классами изображений
model.compile(loss='binary_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])
# Для задачи с несколькими классами изображений
# model.compile(loss='categorical_crossentropy',
#              optimizer='adam', 
#              metrics=['accuracy'])

"""## Обучаем сеть"""

history = model.fit(train_dataset,
                    epochs=10)

"""Проверяем качество обучения на тестовом наборе данных
"""

scores = model.evaluate(test_dataset, verbose=1)


print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 4))


"""## Сохраняем модель"""


model.save("/Models/ml_engineering_weapon_and_no")


"""### Определим функцию для последующей загрузки модели"""

def load_model():
    model = load_model("/Models/ml_engineering_weapon_and_no")
    return model


"""## Использование нейронной сети для распознавания изображений"""


"""Загружаем изображение из файла в StreamLit"""


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


"""Запускаем предобработку и распознавание"""


model = load_model()

st.title('**Классификация оружия на изображении**')
img = load_image()
result = st.button('Распознать изображение')
def print_percent(t):
    return {
               t >= 0.5: str(round(t, 4) * 100),
               t < 0.5: str(round(1 - t, 4) * 100),
           }[1] + "%"


sub = {
   x > 0.5 : "Это оружие",
   x == 0.5: "Не определено",
   x  < 0.5 : "Это НЕ оружие"
}[1]


"""Печатаем результаты распознавания"""


if result:
    x = preprocess_image(img)
    prediction = model.predict(x)
    x = prediction[0][0]
    st.write('**Результаты распознавания: \n **',sub + ", с вероятностью: " + print_percent(x))
   
