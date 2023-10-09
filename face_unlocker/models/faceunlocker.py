import os
import time
import cv2
import shutil
import numpy as np
import tensorflow as tf
from face_unlocker.loaders import load_pretrained_model, load_database


def get_image_from_camera(source=0):
    """This function captures an image from the camera and returns it as a numpy array."""

    cam = cv2.VideoCapture(source)
    time.sleep(1)
    result, image = cam.read()
    if result:
        cv2.imshow('Captured image', image)
        cv2.waitKey(2000)
        cv2.destroyWindow('Captured image')
        cv2.waitKey(1)
        cam.release()
        return image
    else:
        raise Exception('No image detected. Please try again')


class FaceUnlocker:

    def __init__(self, datafolder):
        self.model = load_pretrained_model()
        self.database = load_database(datafolder, self.model)

    def encode(self, image_path):
        """Converts an image to an embedding vector by using the model"""

        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img = np.around(np.array(img) / 255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
        embedding = self.model.predict_on_batch(x_train)
        return embedding / np.linalg.norm(embedding, ord=2)

    def identify_person(self, image_path):
        """Compare the picture from the camera to the pictures in the database"""

        incoming_person_image_encoding = self.encode(image_path)

        distance_between_images = 100

        identified_as = None

        for name, employee_encoding in self.database.items():
            dist = np.linalg.norm(incoming_person_image_encoding - employee_encoding)
            if dist < distance_between_images:
                distance_between_images = dist
                identified_as = name

        if distance_between_images > 0.7:
            print(f'Not sure, maybe it is {identified_as}')
        else:
            print(f'Employee identified\nName: {identified_as}')
            os.system(f"say 'Hello {identified_as}'")

    def recognize_face_from_camera(self):
        """Main function to execute face recognition"""

        face_to_recognize = get_image_from_camera()
        cv2.imwrite('face_to_recognize.jpg', face_to_recognize)
        self.identify_person('face_to_recognize.jpg')
        os.remove('face_to_recognize.jpg')

    def add_new_user_to_database(self, name, source: int | str = 0):
        """Take picture of new employee, store in employees folder and in database as an embedding"""

        if source == 0:
            image = get_image_from_camera()
            image_path = 'face_unlocker/data/' + name + '.jpg'
            cv2.imwrite(image_path, image)
        elif source.endswith(('.jpg', '.png')):
            image_path = source
            shutil.copy(image_path, 'face_unlocker/data/' + name + '.jpg')
        else:
            raise Exception('Invalid source. Please try again')

        if image_path:
            self.database[name] = self.encode(image_path)
            print(f'New user "{name}" added to database')

        return self.database
