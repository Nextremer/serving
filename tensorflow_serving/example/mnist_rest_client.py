# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import numpy as np
import sys
import threading

from flask import abort, Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.example import mnist_input_data

server = 'localhost:9000'

class _Result(object):
  def __init__(self):
    self._prediction = None
    self._scores = None
    self._condition = threading.Condition()
  @property
  def prediction(self): return self_prediction
  @property
  def scores(self): return self._scores
  @prediction.setter
  def prediction(self, value):
    with self._condition:
      self._prediction = value
      self._condition.notify()
  @scores.setter
  def scores(self, value):
    with self._condition:
      self._scores = value
      self._condition.notify()
  def get(self):
    with self._condition:
      while self._prediction is None or self._scores is None:
        self._condition.wait()
      return {'prediction': self._prediction,
              'scores': self._scores.tolist()}

def _create_rpc_callback(result):
  def _callback(result_future):
    result.scores = numpy.array(
      result_future.result().outputs['scores'].float_val)
    result.prediction = numpy.argmax(result.scores)
  return _callback

def do_inference(hostport, image):
  _image = ((255 - np.asarray(image.resize((28, 28)), dtype=np.uint8))
            / 255.0).reshape(1, 784).astype(np.float32)
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'mnist'
  request.model_spec.signature_name = 'predict_images'
  request.inputs['images'].CopyFrom(
    tf.contrib.util.make_tensor_proto(_image, shape=[1, 784]))
  result = _Result()
  result_future = stub.Predict.future(request, 5.0)  # 5 seconds
  result_future.add_done_callback(_create_rpc_callback(result))
  return result.get()

@app.route("/", methods=["POST"])
def hello():
  try:
    image = Image.open(request.files['file'])
  except:
    abort(400)

  result = do_inference(server, image)
  return jsonify(results=result)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8000)
