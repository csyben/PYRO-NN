__author__ = "Christopher Syben <christopher.syben@fau.de>"
__copyright__ = "Christopher Syben <christopher.syben@fau.de>"
__license__ = """
PYRO-NN, python framework for convenient use of the ct reconstructions algorithms
Copyright [2019] [Christopher Syben]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from setuptools import setup, Command
from setuptools.command.build_ext import build_ext
import distutils.command.build as build
import subprocess
import os

ext_mod = []
cmd = {}
try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # class CustomBuildExtension(BuildExtension):
    #     def build_extensions(self):
    #         old_lib = self.build_lib
    #         self.build_lib = os.path.join(self.build_lib, 'pyronn_layers')
    #         super().build_extensions()

    cuda_extension = CUDAExtension(
        name='pyronn_layers_torch',
        sources=[
                    # Python Bindings
                    'src/pyronn/ct_reconstruction/cpp/torch_ops/pyronn_torch_layers.cc',
                    # Parallel operators
                    'src/pyronn/ct_reconstruction/cpp/torch_ops/par_projector_2D_OPKernel.cc',
                    'src/pyronn/ct_reconstruction/cpp/kernels/par_projector_2D_CudaKernel.cu',

                    'src/pyronn/ct_reconstruction/cpp/torch_ops/par_backprojector_2D_OPKernel.cc',
                    'src/pyronn/ct_reconstruction/cpp/kernels/par_backprojector_2D_CudaKernel.cu',
                    # Fan operators
                    'src/pyronn/ct_reconstruction/cpp/torch_ops/fan_projector_2D_OPKernel.cc',
                    'src/pyronn/ct_reconstruction/cpp/kernels/fan_projector_2D_CudaKernel.cu',

                    'src/pyronn/ct_reconstruction/cpp/torch_ops/fan_backprojector_2D_OPKernel.cc',
                    'src/pyronn/ct_reconstruction/cpp/kernels/fan_backprojector_2D_CudaKernel.cu',
                    # Cone operators
                    'src/pyronn/ct_reconstruction/cpp/torch_ops/cone_projector_3D_OPKernel.cc',
                    'src/pyronn/ct_reconstruction/cpp/kernels/cone_projector_3D_CudaKernel.cu',
                    'src/pyronn/ct_reconstruction/cpp/kernels/cone_projector_3D_CudaKernel_hardware_interp.cu',

                    'src/pyronn/ct_reconstruction/cpp/torch_ops/cone_backprojector_3D_OPKernel.cc',
                    'src/pyronn/ct_reconstruction/cpp/kernels/cone_backprojector_3D_CudaKernel.cu',
                    'src/pyronn/ct_reconstruction/cpp/kernels/cone_backprojector_3D_CudaKernel_hardware_interp.cu',
        ],
    )
    ext_mod.append(cuda_extension)
    cmd['build_ext'] = BuildExtension

except Exception as e:
    if isinstance(e, ModuleNotFoundError): pass
    else: raise e

try:
    import tensorflow as tf
    v = tf.__version__.split('.')
    assert int(v[0])>=2 and int(v[1])>=11
    class CustomBuild(build.build):
        def run(self):
            build.build.run(self)

            if not self.dry_run:
                self.make()

        def make(self):
            if os.system("make"):
                raise RuntimeError("Makefile build faied")

            import glob
            import shutil
            so_files = glob.glob('src/pyronn_layers/*.so')
            if not so_files:
                raise RuntimeError('No .so files')
            destination_dir = os.path.join(self.build_lib, 'pyronn_layers')

            os.makedirs(destination_dir, exist_ok=True)
            for so_file in so_files:
                shutil.move(so_file, os.path.join(destination_dir, os.path.basename(so_file)))
                print(f"Moved {so_file} to {destination_dir}")

    cmd['build'] = CustomBuild
except Exception as e:
    if isinstance(e, ModuleNotFoundError): pass
    else: raise e



setup(
   ext_modules=ext_mod,
    include_package_data=True,
    cmdclass=cmd,
)
