import os
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import Imath
import numpy as np
import OpenEXR
from numpy.typing import DTypeLike

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
HALF = Imath.PixelType(Imath.PixelType.HALF)
UINT = Imath.PixelType(Imath.PixelType.UINT)

NO_COMPRESSION = Imath.Compression(Imath.Compression.NO_COMPRESSION)
RLE_COMPRESSION = Imath.Compression(Imath.Compression.RLE_COMPRESSION)
ZIPS_COMPRESSION = Imath.Compression(Imath.Compression.ZIPS_COMPRESSION)
ZIP_COMPRESSION = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
PIZ_COMPRESSION = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
PXR24_COMPRESSION = Imath.Compression(Imath.Compression.PXR24_COMPRESSION)
DWAA_COMPRESSION = Imath.Compression(Imath.Compression.DWAA_COMPRESSION)
DWAB_COMPRESSION = Imath.Compression(Imath.Compression.DWAB_COMPRESSION)

NP_PRECISION: Dict[str, DTypeLike] = {
    "FLOAT": np.float32,
    "HALF": np.float16,
    "UINT": np.uint32,
}


def _compression_level_header_label(compression_type):
    if compression_type == ZIP_COMPRESSION:
        return "zipCompressionLevel"
    elif compression_type == DWAA_COMPRESSION or compression_type == DWAB_COMPRESSION:
        return "dwaCompressionLevel"
    else:
        raise ValueError(f"Not allowed to specify compression level for {compression_type}.")


PathLike = Union[str, bytes, os.PathLike]
PrecisionType = Union[Literal["FLOAT", "HALF", "UINT"], Imath.PixelType]


class ExrError(Exception):
    pass


def open(filename: PathLike) -> "InputFile":
    # Check if the file is an EXR file
    filename = str(filename)
    if not OpenEXR.isOpenExrFile(filename):
        raise ExrError(f"File '{filename}' is not an EXR file.")
    # Return an `InputFile`
    return InputFile(OpenEXR.InputFile(filename))


def read(
    filename: PathLike,
    channels: Union[None, str, Set[str], List[str], Tuple[str, ...]] = "default",
    precision: PrecisionType = FLOAT,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Read and return a numpy array.

    Returns a dictionary of arrays if multiple channels specified.
    """
    with open(str(filename)) as f:
        if _is_list(channels):
            # Construct an array of precisions
            return f.get_dict(channels, precision=precision)
        else:
            return f.get(channels, precision)


def read_all(filename: PathLike, precision: PrecisionType = FLOAT):
    """Read all the channels available."""
    with open(filename) as f:
        return f.get_all(precision=precision)


def _make_ndims_3(matrix: np.ndarray):
    """If necessary, adds a third dimension to 2-dimensional matrices (single channel)."""
    if matrix.ndim > 3 or matrix.ndim < 2:
        raise ValueError("Invalid number of dimensions for the `matrix` argument.")
    elif matrix.ndim == 2:
        matrix = np.expand_dims(matrix, -1)
    return matrix


def _to_type(type_, val):
    return val if isinstance(val, type_) else type_(val)


def _get_channel_names(
    channel_names: Union[None, Dict[str, List[str]], List[str]],
    depth: int,
):
    """Read channel names from default."""
    if channel_names:
        if depth != len(channel_names):
            raise ValueError(
                "The provided channel names have the wrong length (%d vs %d)." % (len(channel_names), depth)
            )
        return channel_names
    elif depth in _default_channel_names:
        return _default_channel_names[depth]
    else:
        raise ValueError("There are no suitable default channel names for data of depth %d" % depth)


def write(
    filename: PathLike,
    data: Union[Dict[str, np.ndarray], np.ndarray],
    channel_names: Optional[Dict[str, List[str]]] = None,
    precision: Union[PrecisionType, Dict[str, PrecisionType]] = FLOAT,
    compression: Imath.Compression = PIZ_COMPRESSION,
    extra_headers: Optional[dict] = None,
    compression_level: Union[None, int, float] = None,
):
    filename = str(filename)

    if extra_headers is None:
        extra_headers = {}

    if compression_level is not None:
        key = _compression_level_header_label(compression)
        extra_headers[key] = compression_level

    if isinstance(data, dict):
        # Make sure everything has ndims 3
        for group, matrix in data.items():
            data[group] = _make_ndims_3(matrix)

        # Prepare precisions
        precisions: Dict[str, Imath.PixelType]
        if isinstance(precision, dict):
            precisions = {group: _to_type(Imath.PixelType, precision.get(group, FLOAT)) for group in data}
        else:
            precisions = {group: _to_type(Imath.PixelType, precision) for group in data}

        # Prepare channel names
        if channel_names is None:
            channel_names = {}

        resolved_channel_names = {
            group: _get_channel_names(channel_names.get(group), matrix.shape[2]) for group, matrix in data.items()
        }

        # Collect channels
        channels, channel_data = {}, {}
        height, width = None, None
        for group, matrix in data.items():
            # Read the depth of the current group
            # and set height and width variables if not set yet
            if width is None:
                height, width, depth = matrix.shape
            else:
                depth = matrix.shape[2]
            names = resolved_channel_names[group]
            # Check the number of channel names
            if len(names) != depth:
                raise ValueError("Depth does not match the number of channel names for channel '%s'" % group)
            for i, c in enumerate(names):
                channel_name = c if group == "default" else f"{group}.{c}"
                channels[channel_name] = Imath.Channel(precisions[group])
                channel_data[channel_name] = matrix[:, :, i].astype(NP_PRECISION[str(precisions[group])]).tobytes()
    elif isinstance(data, np.ndarray):
        data = _make_ndims_3(data)
        height, width, depth = data.shape
        resolved_channel_names = _get_channel_names(channel_names, depth)
        precision = _to_type(Imath.PixelType, precision)
        channels = {c: Imath.Channel(precision) for c in resolved_channel_names}
        channel_data = {
            c: data[:, :, i].astype(NP_PRECISION[str(precision)]).tobytes()
            for i, c in enumerate(resolved_channel_names)
        }
    else:
        raise TypeError("Invalid precision for the `data` argument. Supported are NumPy arrays and dictionaries.")

    # Save
    header = OpenEXR.Header(width, height)
    header.update(extra_headers)
    header["compression"] = compression
    header["channels"] = channels
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels(channel_data)


def tonemap(matrix: np.ndarray, gamma: float = 2.2):
    return np.clip(matrix ** (1.0 / gamma), 0, 1)


class InputFile:
    def __init__(
        self,
        input_file: OpenEXR.InputFile,
    ):
        self.input_file = input_file

        if not input_file.isComplete():
            raise ExrError("EXR file not ready.")

        header = input_file.header()
        dw = header["dataWindow"]

        self.width = dw.max.x - dw.min.x + 1
        self.height = dw.max.y - dw.min.y + 1
        self.channels = sorted(header["channels"].keys(), key=_channel_sort_key)
        self.depth = len(self.channels)
        self.precisions = [c.type for c in header["channels"].values()]
        self.channel_precision = {c: v.type for c, v in header["channels"].items()}
        self.channel_map = defaultdict(list)
        self.root_channels = set()
        self._init_channel_map()

    def _init_channel_map(self):
        # Make a dictionary of subchannels per channel
        for c in self.channels:
            self.channel_map["all"].append(c)
            parts = c.split(".")
            if len(parts) == 1:
                self.root_channels.add("default")
                self.channel_map["default"].append(c)
            else:
                self.root_channels.add(parts[0])
            for i in range(1, len(parts) + 1):
                key = ".".join(parts[0:i])
                self.channel_map[key].append(c)

    def describe_channels(self) -> str:
        """Debugging method."""
        out = []
        for c in self.channel_map.get("default", []):
            out.append(c)
        for group in sorted(self.root_channels):
            if group != "default":
                channels = self.channel_map[group]
                out.append("%-20s%s" % (group, ",".join([c[len(group) + 1 :] for c in channels])))
        return "\n".join(out)

    def get(
        self,
        group: Union[None, str, Set[str], List[str], Tuple[str, ...]] = "default",
        precision: PrecisionType = FLOAT,
    ) -> np.ndarray:
        if group is None:
            group = "default"

        channels = self.channel_map[group]

        if not channels:
            raise ExrError(f"Did not find any channels in group '{group}'.\nTry:\n{self.describe_channels()}")

        strings = self.input_file.channels(channels)

        matrix = np.zeros((self.height, self.width, len(channels)), dtype=NP_PRECISION[str(precision)])
        for i, string in enumerate(strings):
            dtype = NP_PRECISION[str(self.channel_precision[channels[i]])]
            matrix[:, :, i] = np.frombuffer(string, dtype=dtype).reshape(self.height, self.width)
        return matrix

    def get_all(self, precision: Union[None, PrecisionType, Dict[str, PrecisionType]] = None) -> Dict[str, np.ndarray]:
        """Read all the channels available."""
        return self.get_dict(self.root_channels, precision)

    def get_dict(
        self,
        groups: Union[None, str, Set[str], List[str], Tuple[str, ...]] = None,
        precision: Union[None, PrecisionType, Dict[str, PrecisionType]] = None,
    ) -> Dict[str, np.ndarray]:
        if groups is None:
            groups = []

        if precision is None:
            precision = {}

        if not isinstance(precision, dict):
            precision = {group: precision for group in groups}

        out = {}
        channel_items = []
        for group in groups:
            group_chans = self.channel_map[group]
            if len(group_chans) == 0:
                raise ExrError(f"Did not find any channels in group '{group}'.\nTry:\n{self.describe_channels()}")
            p = precision.get(group, FLOAT)
            matrix = np.zeros((self.height, self.width, len(group_chans)), dtype=NP_PRECISION[str(p)])
            out[group] = matrix
            for i, c in enumerate(group_chans):
                channel_items.append({"group": group, "id": i, "channel": c})

        if not channel_items:
            raise ExrError(f"Specify channels; cannot process empty queries.\nTry:\n{self.describe_channels}")

        strings = self.input_file.channels([c["channel"] for c in channel_items])

        for item, s in zip(channel_items, strings):
            dtype = NP_PRECISION[str(self.channel_precision[item["channel"]])]
            out[item["group"]][:, :, item["id"]] = np.frombuffer(s, dtype=dtype).reshape(self.height, self.width)
        return out

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.input_file.close()


_sort_dictionary = {
    "R": "000010",
    "G": "000020",
    "B": "000030",
    "A": "000040",
    "X": "000110",
    "Y": "000120",
    "Z": "000130",
}


def _channel_sort_key(i):
    return [_sort_dictionary.get(x.upper(), x) for x in i.split(".")]


_default_channel_names = {
    1: ["Z"],
    2: ["X", "Y"],
    3: ["R", "G", "B"],
    4: ["R", "G", "B", "A"],
}


def _is_list(x):
    return isinstance(x, (set, list, tuple, np.ndarray))
