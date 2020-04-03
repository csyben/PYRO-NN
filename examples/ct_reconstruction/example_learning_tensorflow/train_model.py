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

import tensorflow as tf
from model import training_parameter as args
from model.geometry_parameters import GEOMETRY
from model.input_data import generate_training_data, generate_validation_data, get_test_data, get_test_cupping_data
from model.pipeline import pipeline
from pyronn.ct_reconstruction.helpers.filters.filters import ram_lak
from plots import evaluation

data, label = generate_training_data()
data_val, label_val = generate_validation_data(args.NUM_VALIDATION_SAMPLES)
data_test, label_test = get_test_data(args.NUM_TEST_SAMPLES)
data_cupping, label_cupping = get_test_cupping_data()


training = pipeline()
training.train(data,label,data_val,label_val)

#Get
initial_filter = training.results.get('initial_filter')
learned_filter = training.results.get('learned_filter')
ram_lak_filter = ram_lak(GEOMETRY.detector_shape, GEOMETRY.detector_spacing)


result_test_initial = training.forward(data_test, label_test, initial_filter)
result_test_ram_lak = training.forward(data_test, label_test, ram_lak_filter)
result_test_learned = training.forward(data_test, label_test, learned_filter)
#
result_cupping_test_initial = training.forward(data_cupping, label_cupping, initial_filter)
result_cupping_test_ram_lak = training.forward(data_cupping, label_cupping, ram_lak_filter)
result_cupping_test_learned = training.forward(data_cupping, label_cupping, learned_filter)


evaluation.evaluation_filter(initial_filter, ram_lak_filter, learned_filter, 'plots/filter.pdf')
evaluation.evaluation_three(result_test_initial, result_test_ram_lak, result_test_learned, GEOMETRY.volume_shape, 'plots/test.pdf')
evaluation.evaluation_three(result_cupping_test_initial, result_cupping_test_ram_lak, result_cupping_test_learned, GEOMETRY.volume_shape, 'plots/cupping.pdf')
