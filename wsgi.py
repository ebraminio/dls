from functools import wraps

import json
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # Xception needs TF

from ipynb.fs.full.dls.core import image_category

from flask import Flask, request, Response
app = Flask(__name__)


def crossorigin(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        response.headers['Access-Control-Allow-Origin'] = "*"
        return response
    return decorated_function


@app.route('/category/<name>')
@crossorigin
def category(name):
    result = image_category(name, request.args.get('model'))
    return Response(json.dumps(result, indent=1, ensure_ascii=False),
                        content_type='application/json;charset=utf8')


@app.route('/')
def maindoc():
    return '''<meta name="robots" content="noindex"><h1>Deep learning services</h1>
Simple tool provides basic deep-learning services for other script.<br>
Current services:
<ul><li>/category/xception/[url]</li></ul>
Source: https://github.com/ebraminio/dls
'''

if __name__ == "__main__":
    app.run()
