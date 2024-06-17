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

datagen = ImageDataGenerator(rescale=1./255,
                             samplewise_center=True,  # set each sample mean to 0
                             rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                             zoom_range=0.1, # Randomly zoom image
                             width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                             height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                             horizontal_flip=True,  # randomly flip images
                             vertical_flip=False)

training_set = datagen.flow_from_directory('../FaceMaskDataset/Train',
                                           target_size= (150,150),
                                           class_mode='binary',
                                           batch_size=8)

validation_set = datagen.flow_from_directory('../FaceMaskDataset/Validation',
                                             target_size= (150,150),
                                             class_mode= 'binary',
                                             batch_size=8)

print(training_set.batch_size)
print(training_set.samples)
print('steps_per_epoch :', training_set.samples/training_set.batch_size)
print('speps_per_epoch :', len(training_set))
print()

print(validation_set.batch_size)
print(validation_set.samples)
print('validation_steps : ', validation_set.samples/validation_set.batch_size)
print('validation_steps : ', len(validation_set))

# load an image from file
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    plt.imshow(img)
    plt.show()

# show_image('../FaceMaskDataset/Test/WithMask/45.png')
# show_image('../FaceMaskDataset/Test/WithoutMask/46.png')


# history = model.fit(
#     training_set,
#     validation_data= validation_set,
#     # steps_per_epoch= len(training_set),
#     # validation_steps=len(validation_set),
#     epochs=10
# )

model.save('Fask_Mask_CNN_Model.h5')
loaded_model = load_model('Fask_Mask_CNN_Model.h5')
loaded_model.summary()

# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(18,5))
# plt.subplot(1,2,1)
# plt.ylim(0, 1.1)
# plt.plot(history['accuracy'])
# plt.plot(history['val_accuracy'])
# plt.legend(['acc', 'val_acc'])
# plt.title('Accuracy')

# plt.subplot(1,2,2)
# plt.ylim(0, 1.5)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['loss', 'val_loss'])
# plt.title('Loss')

# model.save('Fask_Mask_CNN_Model.h5')

# To test for individual images

# mymodel = load_model('../mymodel.h5')
# mymodel.summary()
#
# show_image('../FaceMaskDataset/Test/WithMask/86.png')

# test_image = image.load_img('../FaceMaskDataset/Test/WithMask/86.png',
#                             target_size=(150, 150, 3))
#
# test_image2 = image.load_img('../FaceMaskDataset/Test/WithoutMask/47.png',
#                             target_size=(150, 150, 3))
# # test_image
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = loaded_model.predict(test_image)[0][0]
# # 0 is WithMask, 1 is WithoutMask
# print(result)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.show()

def make_predictions(image_path):
    show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(150, 150))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,150,150,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds

# result1 = make_predictions('../FaceMaskDataset/Test/WithMask/86.png')
# print(result1)
#
# result2 = make_predictions('../FaceMaskDataset/Test/WithoutMask/153.png')
# print(result2)

def classify_Face_Mask(image_path):
    preds = make_predictions(image_path)
    print('preds', preds[0][0])
    if preds[0][0] > 0.5:
        print("FaskMask! Let him in!")
    else:
        print("No FaskMask! Stay out!")

classify_Face_Mask('../FaceMaskDataset/Test/WithMask/86.png')
# classify_Face_Mask('../FaceMaskDataset/Test/WithoutMask/47.png')