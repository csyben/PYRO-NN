__author__ = "Christopher Syben <christopher.syben@fau.de>"
__copyright__ = "Christopher Syben <christopher.syben@fau.de>"
__license__ = """
PYRO-NN, python framework for convenient use of the ct reconstructions algorithms provided within Tensorflow
Copyright (C) 2019 Christopher Syben <christopher.syben@fau.de>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyronn",
    version="0.0.1",
    author="Christopher Syben",
    author_email="christopher.syben@fau.de",
    description="PYRO-NN is the high level Python API to the PYRO-NN-Layers known operators.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csyben/PYRO-NN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
