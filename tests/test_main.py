"""
See test images from their source at:

    https://github.com/AcademySoftwareFoundation/openexr-images/tree/main/TestImages
"""

import pytest

import pyexr


@pytest.fixture
def squares_swirls():
    """
    This image contains colored squares and swirling patterns against a flat gray background. Applying lossy compression algorithms to this image tends to highlight compression artifacts.
    """
    return pyexr.open("tests/openexr-images/TestImages/SquaresSwirls.exr")


@pytest.fixture
def bright_rings_nan_inf():
    """
    This image is the same as BrightRings.exr, except for a few pixels near the center, which contain NaNs and infinities.
    """
    return pyexr.open("tests/openexr-images/TestImages/BrightRingsNanInf.exr")


@pytest.fixture
def bright_rings():
    """
    This RGB image contains a number of rather bright rings, with pixel values over 1000, on a gray background. The image is useful for testing how filtering and resampling algorithms react to high- dynamic-range data. (Some filters, for example, convolution kernels with negative lobes, tend to produce objectionable artifacts near high-contrast edges.) Note that the rings in the image are smooth, although on most displays clamping of the pixel values introduces aliasing artifacts. To see that the rings really are smooth, view the image with and exposure of -10.
    """
    return pyexr.open("tests/openexr-images/TestImages/BrightRings.exr")


@pytest.fixture
def wide_float_range():
    """
    This image contains only a single G channel of type FLOAT. The values in the pixels span almost the entire range of possible 32-bit floating-point numbers (roughly -1e38 to +1e38).
    """
    return pyexr.open("tests/openexr-images/TestImages/WideFloatRange.exr")


@pytest.fixture
def wide_color_gamut():
    """
    Some pixels in this RGB image have extremely saturated colors, outside the gamut that can be displayed on a video monitor whose primaries match Rec. ITU-R BT.709. All RGB triples in the image correspond to CIE xyY triples with xy chromaticities that represent real colors. (In a chromaticity diagram, the pixels are all inside the spectral locus.) However, for pixels whose chromaticities are outside the triangle defined by the chromaticities of the Rec. 709 primaries, at least one of the RGB values is negative.
    """
    return pyexr.open("tests/openexr-images/TestImages/WideColorGamut.exr")


@pytest.fixture
def all_half_values():
    """
    The pixels in this RGB HALF image contain all 65,536 possible 16-bit floating-point numbers, including positive and negative numbers, normalized and denormalized numbers, zero, NaNs, positive infinity and negative infinity.
    """
    return pyexr.open("tests/openexr-images/TestImages/AllHalfValues.exr")


def test_dummy(data):
    assert data == 1
