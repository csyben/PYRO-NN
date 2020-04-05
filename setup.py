__author__ = "Christopher Syben <christopher.syben@fau.de>"
__copyright__ = "Christopher Syben <christopher.syben@fau.de>"
__license__ = """
PYRO-NN, python framework for convenient use of the ct reconstructions algorithms provided within Tensorflow
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
import setuptools

__version__ = '0.1.0'
# REQUIRED_PACKAGES = [
#     'numpy',
#     'pyronn_layers'
# ]

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyronn",
    version=__version__,
    author="Christopher Syben",
    author_email="christopher.syben@fau.de",
    description="PYRO-NN is the high level Python API to the PYRO-NN-Layers known operators.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csyben/PYRO-NN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
