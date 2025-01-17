# IMPLEMENTING LIVE DETECTION OF FACE MASK
import numpy as np
from keras.models import Sequential, load_model
from keras.preprocessing import image
import cv2

mymodel = load_model('Fask_Mask_CNN_Model.h5')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

while cap.isOpened():
    _, img = cap.read()
    face = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    if len(face) !=0:
        for (x, y, w, h) in face:
            face_img = img[y:y + h, x:x + w]
            cv2.imwrite('temp.jpg', face_img)
            test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            pred = mymodel.predict(test_image)[0][0]
            print(pred)
            if pred < 0.5:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(img, 'NO MASK', (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, 'MASK', (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(img, "Face Not Found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow('Face Mask Detector', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
