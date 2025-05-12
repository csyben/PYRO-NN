import pyronn
if pyronn.read_backend() == 'tensorflow':
    import tensorflow as tf
    import os

    _so_file = os.path.join(os.path.dirname(__file__), 'pyronn_layers_tensorflow.so')
    _custom_ops = tf.load_op_library(_so_file)

    for obj in dir(_custom_ops):
        if not obj.startswith('_'):
            globals()[obj] = getattr(_custom_ops, obj)
    print('backend as tensorflow')

elif pyronn.read_backend() == 'torch':
    import torch # DO NOT DELETE THIS!
    import pyronn_layers_torch
    for obj in dir(pyronn_layers_torch):
        if not obj.startswith('_'):
            globals()[obj] = getattr(pyronn_layers_torch, obj)
    print('backend as pytorch')