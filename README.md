# Skin_Disease
# Contains the working code of skin disease classifier using CNN model (MobilenetV2).
# still under optimization 
import tensorflow as tf    #importing necessary libraries
  
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2

train_dir = '/content/drive/MyDrive/train_set'
validation_dir = '/content/drive/MyDrive/test_set'

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/train_set',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)

validation_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/test_set (1)',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)

%tensorflow_version 2.x
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import platform

model = tf.keras.applications.MobileNetV2()
print(model)

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/train_set',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)

validation_generator = datagen.flow_from_directory(
    '/content/drive/MyDrive/test_set (1)',
    target_size=(224, 224),
    batch_size=4,
    class_mode='categorical',
    subset='validation',
)

base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

fine_tune_at = 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, validation_data=validation_generator, epochs=25)

class_names = ['cellulitis', 'impetigo', 'athlete-foot','nail-fungus','ring worm','cutaneous-larva-migrans','chickenpox','shingles']

import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

test_image = load_and_preprocess_image('/content/drive/MyDrive/testing 1.jpg')  # Implement this function

# Get predictions
predictions = model.predict(np.expand_dims(test_image, axis=0))
predicted_class_index = np.argmax(predictions)
predicted_class_name = class_names[predicted_class_index]

print(f"Predicted class: {predicted_class_name}")
test_image = load_and_preprocess_image('/content/drive/MyDrive/testing 1.jpg')  # Implement this function



