#!/bin/bash -e

cd /serving
nohup bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/ --logtostderr &
bazel-bin/tensorflow_serving/example/mnist_rest_client
