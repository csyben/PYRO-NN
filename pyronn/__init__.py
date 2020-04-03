# Copyright [2019] [Christopher Syben, Markus Michen]
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

try:
    import pyronn_layers
    import tensorflow as tf
    try:
        print("Tensorflow: memory_growth is set to True")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RunetimeError as e:
                print(e)
    except error as e:
        print("Failed to allow memory growth for Tensorflow. Be aware of memory limitations using the hardware interpolate flag for the PYRO-NN layers.")
except ImportError:
    import warnings
    warnings.warn('Could not import PYRO-NN-Layers. Please install PYRO-NN-Layers from www.github.com/csyben/pyRO-NN-Layers')

name = "pyronn"