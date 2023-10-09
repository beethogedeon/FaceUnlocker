from tensorflow.keras.models import model_from_json
from face_unlocker.encoders import img_to_encoding
import os


def load_pretrained_model():
    """Load the saved model from the disk"""

    json_file = open('../models/pretrained/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('../models/pretrained/model.h5')
    return model


def load_database(directory, model):
    """Initialize the database of people names and their photos encodings"""
    database = {}
    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            image_path = os.path.join(directory, file)
            embedding = img_to_encoding(image_path, model)
            database[file[:-4]] = embedding
    return database
