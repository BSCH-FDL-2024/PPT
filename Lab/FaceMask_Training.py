import numpy as np
import tensorflow
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from keras.models import Sequential, load_model
from keras.preprocessing import image
import cv2

# UNCOMMENT THE FOLLOWING CODE TO TRAIN THE CNN FROM SCRATCH
# BUILDING MODEL TO CLASSIFY BETWEEN MASK AND NO MASK

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    '../FaceMaskDataset/Train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    '../FaceMaskDataset/Test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

# model_saved = model.fit(
#     training_set,
#     epochs=10,
#     validation_data=test_set,
# )

# model.save('mymodel.h5', model_saved)

# To test for individual images

mymodel = load_model('../mymodel.h5')
mymodel.summary()

test_image = image.load_img('../FaceMaskDataset/Test/WithMask/86.png',
                            target_size=(150, 150, 3))

test_image2 = image.load_img('../FaceMaskDataset/Test/WithoutMask/47.png',
                            target_size=(150, 150, 3))
# test_image
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = mymodel.predict(test_image)[0][0]
# 0 is WithMask, 1 is WithoutMask
print(result)