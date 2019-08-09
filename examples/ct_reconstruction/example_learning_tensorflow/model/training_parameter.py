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

# training parameters
LEARNING_RATE          = 1e-6
BATCH_SIZE_TRAIN       = 1
NUM_TRAINING_SAMPLES   = 10
MAX_TRAIN_STEPS        = NUM_TRAINING_SAMPLES//BATCH_SIZE_TRAIN +1
BATCH_SIZE_VALIDATION  = 1
NUM_VALIDATION_SAMPLES = 10
MAX_VALIDATION_STEPS   = NUM_VALIDATION_SAMPLES//BATCH_SIZE_VALIDATION
NUM_TEST_SAMPLES       = 1
MAX_TEST_STEPS         = NUM_TEST_SAMPLES
MAX_EPOCHS             = 100
#Path
LOG_DIR = 'logs/'
WEIGHTS_DIR = 'trained_models/'