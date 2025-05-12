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
import os
import json
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'CONFIG.json')

def default_config():
    config = {'backend': 'torch'}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def read_backend():
    if not os.path.exists(CONFIG_FILE):
        default_config()
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    return config['backend']

def set_backend(value):
    config = {'backend': value}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

name = "pyronn"