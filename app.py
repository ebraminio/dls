import os
os.environ["KERAS_BACKEND"] = "tensorflow" # Xception needs TF

import io
import json
import re
import urllib.parse
from functools import wraps

from flask import Flask, request, Response
import requests

import numpy as np
from PIL import Image

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import MobileNet
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
import keras.applications.inception_v3

from open_nsfw import OpenNsfw


app = Flask(__name__)

MODELS = {
    "mobilenet": MobileNet,
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception, # TensorFlow ONLY
    "resnet": ResNet50
}

available_models = {}
#available_models = {model_name: model(weights="imagenet") for model_name, model in MODELS.items()}


def img_classify(img, model_name="xception"):
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input

    if model_name in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = keras.applications.inception_v3.preprocess_input

    if model_name not in available_models:
        available_models[model_name] = MODELS[model_name](weights="imagenet")
    model = available_models[model_name]

    img = img_to_array(img.convert('RGB').resize(inputShape))
    # prepend a numpy array dimension, (x, y, 3) -> (1, x, y, 3)
    img = preprocess(np.expand_dims(img, axis=0))

    preds = model.predict(img)
    P = imagenet_utils.decode_predictions(preds)

    return [[label, float(prob), imagenetID] for (imagenetID, label, prob) in P[0]]


def crossorigin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        response.headers['Access-Control-Allow-Origin'] = "*"
        return response
    return decorated_function


@app.route('/classify/<model>/<path:url>')
@crossorigin
def classify(model, url):
    if not url.startswith('https://upload.wikimedia.org/'):
        if not url.startswith('https://commons.wikimedia.org/'):
            raise Exception('Only Wikipedia images are supported for now')

    if model not in MODELS:
        raise Exception('Requested model not available')

    result = img_classify(Image.open(io.BytesIO(requests.get(url).content)))
    return Response(json.dumps(result, indent=1, ensure_ascii=False),
                        content_type='application/json;charset=utf8')


def img_nsfw(img):
    img = img_to_array(img.convert('RGB').resize((224, 224)))
    # prepend a numpy array dimension, (x, y, 3) -> (1, x, y, 3)
    img = np.expand_dims(img, axis=0)
    img = imagenet_utils.preprocess_input(img)

    if 'nsfw' not in available_models:
        available_models['nsfw'] = OpenNsfw(include_top=True, weights='yahoo')

    preds = available_models['nsfw'].predict(img)
    
    return preds[0, 1]


@app.route('/nsfw/<path:url>')
@crossorigin
def nsfw(url):
    if not url.startswith('https://upload.wikimedia.org/'):
        if not url.startswith('https://commons.wikimedia.org/'):
            raise Exception('Only Wikipedia images are supported for now')

    result = {
        'result': float(img_nsfw(Image.open(io.BytesIO(requests.get(url).content))))
    }
    return Response(json.dumps(result, indent=1, ensure_ascii=False),
                    content_type='application/json;charset=utf8')


@app.route('/')
def maindoc():
    return '''<meta name="robots" content="noindex"><h1>Deep learning services</h1>
Simple tool provides basic deep-learning services for other script.<br>
Current services:
<ul><li>/deep-learning-services/classify/xception/[url]</li><li>/deep-learning-services/nsfw/[url]</li></ul>
Source: https://github.com/ebraminio/dls
'''


if __name__ == '__main__':
    app.run()
