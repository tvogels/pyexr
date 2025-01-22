"""Test pyexr core capabilities.

See EXR test images from their source at:

    https://github.com/AcademySoftwareFoundation/openexr-images/tree/main/TestImages
"""

import numpy as np
import pytest

from pathlib import Path

import pyexr


@pytest.fixture
def squares_swirls():
    """
    This image contains colored squares and swirling patterns against a flat gray background. Applying lossy compression algorithms to this image tends to highlight compression artifacts.
    """
    with pyexr.open("tests/openexr-images/TestImages/SquaresSwirls.exr") as f:
        yield f


@pytest.fixture
def bright_rings_nan_inf():
    """
    This image is the same as BrightRings.exr, except for a few pixels near the center, which contain NaNs and infinities.
    """
    with pyexr.open("tests/openexr-images/TestImages/BrightRingsNanInf.exr") as f:
        yield f


@pytest.fixture
def bright_rings():
    """
    This RGB image contains a number of rather bright rings, with pixel values over 1000, on a gray background. The image is useful for testing how filtering and resampling algorithms react to high- dynamic-range data. (Some filters, for example, convolution kernels with negative lobes, tend to produce objectionable artifacts near high-contrast edges.) Note that the rings in the image are smooth, although on most displays clamping of the pixel values introduces aliasing artifacts. To see that the rings really are smooth, view the image with and exposure of -10.
    """
    with pyexr.open("tests/openexr-images/TestImages/BrightRings.exr") as f:
        yield f


@pytest.fixture
def wide_float_range():
    """
    This image contains only a single G channel of type FLOAT. The values in the pixels span almost the entire range of possible 32-bit floating-point numbers (roughly -1e38 to +1e38).
    """
    with pyexr.open("tests/openexr-images/TestImages/WideFloatRange.exr") as f:
        yield f


@pytest.fixture
def wide_color_gamut():
    """
    Some pixels in this RGB image have extremely saturated colors, outside the gamut that can be displayed on a video monitor whose primaries match Rec. ITU-R BT.709. All RGB triples in the image correspond to CIE xyY triples with xy chromaticities that represent real colors. (In a chromaticity diagram, the pixels are all inside the spectral locus.) However, for pixels whose chromaticities are outside the triangle defined by the chromaticities of the Rec. 709 primaries, at least one of the RGB values is negative.
    """
    with pyexr.open("tests/openexr-images/TestImages/WideColorGamut.exr") as f:
        yield f


@pytest.fixture
def all_half_values():
    """
    The pixels in this RGB HALF image contain all 65,536 possible 16-bit floating-point numbers, including positive and negative numbers, normalized and denormalized numbers, zero, NaNs, positive infinity and negative infinity.
    """
    with pyexr.open("tests/openexr-images/TestImages/AllHalfValues.exr") as f:
        yield f


@pytest.mark.parametrize(
    "path",
    [
        "tests/openexr-images/TestImages/AllHalfValues.exr",
        Path("tests/openexr-images/TestImages/AllHalfValues.exr"),
    ],
)
def test_read_direct_default(path):
    """Tests both ``str`` and ``Path`` inputs."""
    data = pyexr.read(path)

    assert isinstance(data, np.ndarray)
    assert data.shape == (256, 256, 3)
    assert data.dtype == np.float32


def test_read_direct_channels():
    data = pyexr.read("tests/openexr-images/TestImages/AllHalfValues.exr", channels=["R"])

    assert isinstance(data, dict)
    assert data["R"].shape == (256, 256, 1)
    assert data["R"].dtype == np.float32


def test_read_all_half_values(all_half_values):
    assert all_half_values.channels == ["R", "G", "B"]
    assert all_half_values.width == 256
    assert all_half_values.height == 256
    assert all_half_values.channel_precision == {
        "R": pyexr.HALF,
        "G": pyexr.HALF,
        "B": pyexr.HALF,
    }

    img = all_half_values.get()
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.float32


def test_read_all_half_values_single_channel(all_half_values):
    img = all_half_values.get("R")
    assert img.shape == (256, 256, 1)
    assert img.dtype == np.float32


def test_read_all_half_values_explicit_precision(all_half_values):
    img = all_half_values.get(precision=pyexr.HALF)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.float16


@pytest.fixture
def depth():
    return np.ones((480, 640))


@pytest.fixture
def flow():
    return np.ones((480, 640, 2))


@pytest.fixture
def rgb():
    return np.ones((480, 640, 3))


@pytest.fixture
def rgba():
    return np.ones((480, 640, 4))


@pytest.fixture
def fn(tmp_path):
    return tmp_path / "out.exr"


def test_write_single_channel_default(fn, depth):
    pyexr.write(fn, depth)
    with pyexr.open(fn) as f:
        assert f.channels == ["Z"]
        assert f.width == 640
        assert f.height == 480
        assert f.channel_precision == {
            "Z": pyexr.FLOAT,
        }


def test_write_single_channel_dwaa_compression(fn, depth):
    pyexr.write(fn, depth, compression=pyexr.DWAA_COMPRESSION, compression_level=50)

    with pyexr.open(fn) as f:
        assert f.channels == ["Z"]
        assert f.width == 640
        assert f.height == 480
        assert f.channel_precision == {
            "Z": pyexr.FLOAT,
        }
    readback = pyexr.read_all(fn)
    np.testing.assert_allclose(depth, readback["default"][..., 0])


def test_write_custom_and_read_all(fn, depth, rgb):
    data = {"default": rgb, "Depth": depth}
    pyexr.write(fn, data, precision={"default": pyexr.HALF}, channel_names={"Depth": ["Q"]})

    with pyexr.open(fn) as f:
        assert f.channels == ["R", "G", "B", "Depth.Q"]
        assert f.width == 640
        assert f.height == 480
        assert f.channel_precision == {
            "B": pyexr.HALF,
            "Depth.Q": pyexr.FLOAT,
            "G": pyexr.HALF,
            "R": pyexr.HALF,
        }

    readback = pyexr.read_all(fn)
    assert readback["default"].shape == (480, 640, 3)
    assert readback["Depth"].shape == (480, 640, 1)


def test_describe_channels_custom(fn, depth, rgb):
    data = {"default": rgb, "Depth": depth}
    pyexr.write(fn, data, channel_names={"Depth": ["Q"]})
    with pyexr.open(fn) as f:
        actual = f.describe_channels()
        assert actual == "R\nG\nB\nDepth               Q"


def test_describe_channels_default(all_half_values):
    actual = all_half_values.describe_channels()
    assert actual == "R\nG\nB"
