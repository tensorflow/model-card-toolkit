# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Utilities for reading and writing files."""

import os
from pathlib import Path
from typing import Union

from google.protobuf.message import Message


def suffix(path: Union[str, Path]) -> str:
  """Returns the suffix (extension) of a path."""
  return os.path.splitext(path)[1]


def write_file(
    path: Union[str, Path], content: Union[str, bytes], mode: str = 'w'
):
  """Writes content to a file, creating any necessary directories.

  Args:
    path: The file path to write to
    content: The content to write.
    mode: The mode to open the file in. Defaults to 'w'.

  Raises:
    ValueError: If an invalid mode is provided.
  """
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, mode) as f:
    f.write(content)


def write_proto_file(path: Union[str, Path], proto: Message):
  """Writes a message to a proto file, creating any necessary directories.

  Args:
    path: The file path to write the proto to.
    proto: The proto to write.
  """
  write_file(path, proto.SerializeToString(), mode='wb')


def read_file(path: Union[str, Path], mode: str = 'r') -> Union[str, bytes]:
  """Reads a file and returns its content.

  Args:
    path: The file path to read from.
    mode: The mode to open the file in. Defaults to 'r'.

  Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: If an invalid mode is provided.
  """
  if not os.path.exists(path):
    raise FileNotFoundError(f'File {path} does not exist.')

  with open(path, mode) as f:
    return f.read()


def parse_proto_file(path: Union[str, Path], proto: Message) -> Message:
  """Parses a message from a proto file and returns the proto.

  Args:
    path: The file path to parse the proto from.
    proto: The proto to parse into.

  Raises:
    FileNotFoundError: If the file does not exist.
  """
  if not os.path.exists(path):
    raise FileNotFoundError(f'File {path} does not exist.')

  with open(path, 'rb') as f:
    proto.ParseFromString(f.read())
  return proto
