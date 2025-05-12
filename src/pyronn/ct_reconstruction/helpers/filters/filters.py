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

import numpy as np
from typing import Tuple


def ramp(detector_width:int)->np.array:
    """
    create a 1d ramp filter.

    :param detector_width: width of detector(filter)
    :return: filter
    """
    filter_array = np.zeros(detector_width)
    frequency_spacing = 0.5 / (detector_width / 2.0)
    for i in range(0, filter_array.shape[0]):
        if i <= filter_array.shape[0] / 2.0:
            filter_array[i] = i * frequency_spacing
        elif i > filter_array.shape[0] / 2.0:
            filter_array[i] = 0.5 - (((i - filter_array.shape[0] / 2.0)) * frequency_spacing)
    return filter_array.astype(np.float32)


def ramp_2D(detector_shape:Tuple[int,int], number_of_projections:int)->np.array:
    """
    create a 2d ramp filter.

    :param detector_shape: shape of detector
    :param number_of_projections: number of projections
    :return: a 2d ramp filter
    """
    detector_width = detector_shape[-1]

    filter = [
        np.reshape(
            ramp(detector_width),
            (1, detector_width)
        )
        for i in range(0, number_of_projections)
    ]

    filter = np.concatenate(filter)

    return filter


def ramp_3D(detector_shape:Tuple[int,int,int], number_of_projections:int)->np.array:
    """
    create a 3d ramp filter

    :param detector_shape: shape of detector
    :param number_of_projections: number of projections
    :return: a 3d ramp filter
    """
    detector_width = detector_shape[-1]

    filter = [
        np.reshape(
            ramp(detector_width),
            (1, 1, detector_width)
        )
        for i in range(0, number_of_projections)
    ]

    filter = np.concatenate(filter)

    return filter


# Yipeng rewrote the filters direct in frequency domain
def ram_lak(num_detectors: int, detector_spacing: float) -> np.array:
    """Generate the RAM-LAK (Ramp) filter in the frequency domain."""
    frequencies = np.fft.fftfreq(num_detectors)
    ramp = 1.0 / (detector_spacing * detector_spacing)
    filter = ramp * np.abs(frequencies)

    return filter.astype(np.float32)


def ram_lak_2D(
    detector_shape: Tuple[int, int],
    detector_spacing: Tuple[float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = ram_lak(detector_width, detector_spacing_width)
    filter_2D = np.tile(filter_1D, (number_of_projections, 1))

    return filter_2D


def ram_lak_3D(
    detector_shape: Tuple[int, int, int],
    detector_spacing: Tuple[float, float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = ram_lak(detector_width, detector_spacing_width)
    filter_3D = np.tile(
        filter_1D.reshape(1, 1, detector_width), (number_of_projections, 1, 1)
    )

    return filter_3D


def shepp_logan(num_detectors: int, detector_spacing: float) -> np.array:
    """Generate the Shepp-Logan filter in the frequency domain."""
    frequencies = np.fft.fftfreq(num_detectors)
    ramp = 1.0 / (detector_spacing * detector_spacing)
    filter = ramp * np.abs(frequencies)
    sinc_filter = np.where(
        frequencies == 0, 1.0, np.sin(np.pi * frequencies) / (np.pi * frequencies)
    )

    return (filter * sinc_filter).astype(np.float32)


def shepp_logan_2D(
    detector_shape: Tuple[int, int],
    detector_spacing: Tuple[float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = shepp_logan(detector_width, detector_spacing_width)
    filter_2D = np.tile(filter_1D, (number_of_projections, 1))

    return filter_2D


def shepp_logan_3D(
    detector_shape: Tuple[int, int, int],
    detector_spacing: Tuple[float, float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = shepp_logan(detector_width, detector_spacing_width)
    filter_3D = np.tile(
        filter_1D.reshape(1, 1, detector_width), (number_of_projections, 1, 1)
    )

    return filter_3D


def cosine(num_detectors: int, detector_spacing: float) -> np.array:
    """Generate the Cosine filter in the frequency domain."""
    frequencies = np.fft.fftfreq(num_detectors)
    ramp = 1.0 / (detector_spacing * detector_spacing)
    filter = ramp * np.abs(frequencies) * np.cos(np.pi * frequencies / 2)

    return filter.astype(np.float32)


def cosine_2D(
    detector_shape: Tuple[int, int],
    detector_spacing: Tuple[float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = cosine(detector_width, detector_spacing_width)
    filter_2D = np.tile(filter_1D, (number_of_projections, 1))

    return filter_2D


def cosine_3D(
    detector_shape: Tuple[int, int, int],
    detector_spacing: Tuple[float, float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = cosine(detector_width, detector_spacing_width)
    filter_3D = np.tile(
        filter_1D.reshape(1, 1, detector_width), (number_of_projections, 1, 1)
    )

    return filter_3D


def hamming(num_detectors: int, detector_spacing: float) -> np.array:
    """Generate the Hamming filter in the frequency domain."""
    frequencies = np.fft.fftfreq(num_detectors)
    ramp = 1.0 / (detector_spacing * detector_spacing)
    filter = ramp * np.abs(frequencies) * (0.54 + 0.46 * np.cos(np.pi * frequencies))

    return filter.astype(np.float32)


def hamming_2D(
    detector_shape: Tuple[int, int],
    detector_spacing: Tuple[float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = hamming(detector_width, detector_spacing_width)
    filter_2D = np.tile(filter_1D, (number_of_projections, 1))

    return filter_2D


def hamming_3D(
    detector_shape: Tuple[int, int, int],
    detector_spacing: Tuple[float, float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = hamming(detector_width, detector_spacing_width)
    filter_3D = np.tile(
        filter_1D.reshape(1, 1, detector_width), (number_of_projections, 1, 1)
    )

    return filter_3D


def hann(num_detectors: int, detector_spacing: float) -> np.array:
    """Generate the Hann filter in the frequency domain."""
    frequencies = np.fft.fftfreq(num_detectors)
    ramp = 1.0 / (detector_spacing * detector_spacing)
    filter = ramp * np.abs(frequencies) * (0.5 + 0.5 * np.cos(np.pi * frequencies))

    return filter.astype(np.float32)


def hann_2D(
    detector_shape: Tuple[int, int],
    detector_spacing: Tuple[float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = hann(detector_width, detector_spacing_width)
    filter_2D = np.tile(filter_1D, (number_of_projections, 1))

    return filter_2D


def hann_3D(
    detector_shape: Tuple[int, int, int],
    detector_spacing: Tuple[float, float, float],
    number_of_projections: int,
) -> np.array:
    detector_width = detector_shape[-1]
    detector_spacing_width = detector_spacing[-1]

    filter_1D = hann(detector_width, detector_spacing_width)
    filter_3D = np.tile(
        filter_1D.reshape(1, 1, detector_width), (number_of_projections, 1, 1)
    )

    return filter_3D
