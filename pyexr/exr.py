from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *


import OpenEXR, Imath
import numpy as np
from collections import defaultdict
import os, sys

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
HALF  = Imath.PixelType(Imath.PixelType.HALF)
UINT  = Imath.PixelType(Imath.PixelType.UINT)

NO_COMPRESSION    = Imath.Compression(Imath.Compression.NO_COMPRESSION)
RLE_COMPRESSION   = Imath.Compression(Imath.Compression.RLE_COMPRESSION)
ZIPS_COMPRESSION  = Imath.Compression(Imath.Compression.ZIPS_COMPRESSION)
ZIP_COMPRESSION   = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
PIZ_COMPRESSION   = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
PXR24_COMPRESSION = Imath.Compression(Imath.Compression.PXR24_COMPRESSION)

NP_PRECISION = {
  "FLOAT": np.float32,
  "HALF":  np.float16,
  "UINT":  np.uint8
}


def open(filename):
  # Check if the file is an EXR file
  if not OpenEXR.isOpenExrFile(filename):
    raise Exception("File '%s' is not an EXR file." % filename)
  # Return an `InputFile`
  return InputFile(OpenEXR.InputFile(filename), filename)


def read(filename, channels = "default", precision = FLOAT):
  f = open(filename)
  if _is_list(channels):
    # Construct an array of precisions
    return f.get_dict(channels, precision=precision)

  else:
    return f.get(channels, precision)

def read_all(filename, precision = FLOAT):
  f = open(filename)
  return f.get_all(precision=precision)

def write(filename, data, channel_names = None, precision = FLOAT, compression = PIZ_COMPRESSION):

  # Helper function add a third dimension to 2-dimensional matrices (single channel)
  def make_ndims_3(matrix):
    if matrix.ndim > 3 or matrix.ndim < 2:
      raise Exception("Invalid number of dimensions for the `matrix` argument.")
    elif matrix.ndim is 2:
      matrix = np.expand_dims(matrix, -1)
    return matrix

  # Helper function to read channel names from default
  def get_channel_names(channel_names, depth):
    if channel_names:
      if depth is not len(channel_names):
        raise Exception("The provided channel names have the wrong length (%d vs %d)." % (len(channel_names), depth))
      return channel_names
    elif depth in _default_channel_names:
      return _default_channel_names[depth]
    else:
      raise Exception("There are no suitable default channel names for data of depth %d" % depth)

  #
  # Case 1, the `data` argument is a dictionary
  #
  if isinstance(data, dict):
    # Make sure everything has ndims 3
    for group, matrix in data.items():
      data[group] = make_ndims_3(matrix)

    # Prepare precisions
    if not isinstance(precision, dict):
      precisions = {group: precision for group in data.keys()}
    else:
      precisions = {group: precision.get(group, FLOAT) for group in data.keys()}

    # Prepare channel names
    if channel_names is None:
      channel_names = {}
    channel_names = {group: get_channel_names(channel_names.get(group), matrix.shape[2]) for group, matrix in data.items()}

    # Collect channels
    channels = {}
    channel_data = {}
    width = None
    height = None
    for group, matrix in data.items():
      # Read the depth of the current group
      # and set height and width variables if not set yet
      if width is None:
        height, width, depth = matrix.shape
      else:
        depth = matrix.shape[2]
      names = channel_names[group]
      # Check the number of channel names
      if len(names) != depth:
        raise Exception("Depth does not match the number of channel names for channel '%s'" % group)
      for i, c in enumerate(names):
        if group == "default":
          channel_name = c
        else:
          channel_name = "%s.%s" % (group, c)
        channels[channel_name] = Imath.Channel(precisions[group])
        channel_data[channel_name] = matrix[:,:,i].astype(NP_PRECISION[str(precisions[group])]).tostring()

    # Save
    header = OpenEXR.Header(width, height)
    header['compression'] = compression
    header['channels'] = channels
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels(channel_data)

  #
  # Case 2, the `data` argument is one matrix
  #
  elif isinstance(data, np.ndarray):
    data = make_ndims_3(data)
    height, width, depth = data.shape
    channel_names = get_channel_names(channel_names, depth)
    header = OpenEXR.Header(width, height)
    header['compression'] = compression
    header['channels'] = {c: Imath.Channel(precision) for c in channel_names}
    out = OpenEXR.OutputFile(filename, header)
    out.writePixels({c: data[:,:,i].astype(NP_PRECISION[str(precision)]).tostring() for i, c in enumerate(channel_names)})

  else:
    raise Exception("Invalid precision for the `data` argument. Supported are NumPy arrays and dictionaries.")


def tonemap(matrix, gamma=2.2):
  return np.clip(matrix ** (1.0/gamma), 0, 1)


class InputFile(object):

  def __init__(self, input_file, filename=None):
    self.input_file = input_file

    if not input_file.isComplete():
      raise Exception("EXR file '%s' is not ready." % filename)

    header = input_file.header()
    dw     = header['dataWindow']

    self.width             = dw.max.x - dw.min.x + 1
    self.height            = dw.max.y - dw.min.y + 1
    self.channels          = sorted(header['channels'].keys(),key=_channel_sort_key)
    self.depth             = len(self.channels)
    self.precisions        = [c.type for c in header['channels'].values()]
    self.channel_precision = {c: v.type for c, v in header['channels'].items()}
    self.channel_map       = defaultdict(list)
    self.root_channels     = set()
    self._init_channel_map()

  def _init_channel_map(self):
    # Make a dictionary of subchannels per channel
    for c in self.channels:
      self.channel_map['all'].append(c)
      parts = c.split('.')
      if len(parts) is 1:
        self.root_channels.add('default')
        self.channel_map['default'].append(c)
      else:
        self.root_channels.add(parts[0])
      for i in range(1, len(parts)+1):
        key = ".".join(parts[0:i])
        self.channel_map[key].append(c)

  def describe_channels(self):
    if 'default' in self.root_channels:
      for c in self.channel_map['default']:
        print (c)
    for group in sorted(list(self.root_channels)):
      if group != 'default':
        channels = self.channel_map[group]
        print("%-20s%s" % (group, ",".join([c[len(group)+1:] for c in channels])))

  def get(self, group = 'default', precision=FLOAT):
    channels = self.channel_map[group]

    if len(channels) == 0:
      print("I did't find any channels in group '%s'." % group)
      print("You could try:")
      self.describe_channels()
      sys.exit()

    strings = self.input_file.channels(channels)

    matrix = np.zeros((self.height, self.width, len(channels)), dtype=NP_PRECISION[str(precision)])
    for i, string in enumerate(strings):
      precision = NP_PRECISION[str(self.channel_precision[channels[i]])]
      matrix[:,:,i] = np.fromstring(string, dtype = precision) \
                        .reshape(self.height, self.width)
    return matrix

  def get_all(self, precision = {}):
    return self.get_dict(self.root_channels, precision)

  def get_dict(self, groups = [], precision = {}):

    if not isinstance(precision, dict):
      precision = {group: precision for group in groups}

    return_dict = {}
    todo = []
    for group in groups:
      group_chans = self.channel_map[group]
      if len(group_chans) == 0:
        print("I didn't find any channels for the requested group '%s'." % group)
        print("You could try:")
        self.describe_channels()
        sys.exit()
      if group in precision:
        p = precision[group]
      else:
        p = FLOAT
      matrix = np.zeros((self.height, self.width, len(group_chans)), dtype=NP_PRECISION[str(p)])
      return_dict[group] = matrix
      for i, c in enumerate(group_chans):
        todo.append({'group': group, 'id': i, 'channel': c})

    if len(todo) == 0:
      print("Please ask for some channels, I cannot process empty queries.")
      print("You could try:")
      self.describe_channels()
      sys.exit()

    strings = self.input_file.channels([c['channel'] for c in todo])

    for i, item in enumerate(todo):
      precision = NP_PRECISION[str(self.channel_precision[todo[i]['channel']])]
      return_dict[item['group']][:,:,item['id']] = \
          np.fromstring(strings[i], dtype = precision) \
            .reshape(self.height, self.width)
    return return_dict


def _sort_dictionary(key):
  if key == 'R' or key == 'r':
    return "000010"
  elif key == 'G' or key == 'g':
    return "000020"
  elif key == 'B' or key == 'b':
    return "000030"
  elif key == 'A' or key == 'a':
    return "000040"
  elif key == 'X' or key == 'x':
    return "000110"
  elif key == 'Y' or key == 'y':
    return "000120"
  elif key == 'Z' or key == 'z':
    return "000130"
  else:
    return key


def _channel_sort_key(i):
  return [_sort_dictionary(x) for x in i.split(".")]


_default_channel_names = {
  1: ['Z'],
  2: ['X','Y'],
  3: ['R','G','B'],
  4: ['R','G','B','A']
}


def _is_list(x):
  return isinstance(x, (list, tuple, np.ndarray))







