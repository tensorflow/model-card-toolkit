# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BaseModelCardField.

This class serves as a basic shared API between all Model Card data classes (
see model_card.py).
"""

import abc
import dataclasses
import json as json_lib
from typing import Any, Dict

from google.protobuf import descriptor, message

from model_card_toolkit.utils import validation


class BaseModelCardField(abc.ABC):
  """Model card field base class.

  This is an abstract class. All the model card fields should inherit this class
  and override the _proto_type property to the corresponding proto type. This
  abstract class provides methods `copy_from_proto`, `merge_from_proto` and
  `to_proto` to convert the class from and to proto. The child class does not
  need to override this unless it needs some special process.
  """
  def __len__(self) -> int:
    """Returns the number of items in a field. Ignores None values recursively,
    so the length of a field that only contains another field that has all None
    values would be 0.
    """
    return len(self.to_dict())

  @property
  @abc.abstractmethod
  def _proto_type(self):
    """The proto type. Child class should overwrite this."""

  def to_proto(self) -> message.Message:
    """Convert this class object to the proto."""
    proto = self._proto_type()

    for field_name, field_value in self.__dict__.items():
      if not hasattr(proto, field_name):
        raise ValueError(
            "%s has no such field named '%s'." % (type(proto), field_name)
        )
      if not field_value:
        continue

      field_descriptor = proto.DESCRIPTOR.fields_by_name[field_name]

      # Process Message type.
      if field_descriptor.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
        if field_descriptor.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          for nested_message in field_value:
            getattr(proto, field_name).add().CopyFrom(
                nested_message.to_proto()
            )  # pylint: disable=protected-access
        else:
          getattr(proto, field_name).CopyFrom(field_value.to_proto())  # pylint: disable=protected-access
      # Process Non-Message type
      else:
        if field_descriptor.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          getattr(proto, field_name).extend(field_value)
        else:
          setattr(proto, field_name, field_value)

    return proto

  def _from_proto(self, proto: message.Message) -> "BaseModelCardField":
    """Convert proto to this class object."""
    if not isinstance(proto, self._proto_type):
      raise TypeError(
          "%s is expected. However %s is provided." %
          (self._proto_type, type(proto))
      )

    for field_descriptor in proto.DESCRIPTOR.fields:
      field_name = field_descriptor.name
      if not hasattr(self, field_name):
        raise ValueError(
            "%s has no such field named '%s.'" % (self, field_name)
        )

      # Process Message type.
      if field_descriptor.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
        if field_descriptor.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          # Clean the list first.
          setattr(self, field_name, [])
          for p in getattr(proto, field_name):
            # To get the type hint of a list is not easy.
            field = \
              self.__annotations__[field_name].__args__[0]()  # pytype: disable=attribute-error
            field._from_proto(p)  # pylint: disable=protected-access
            getattr(self, field_name).append(field)

        elif proto.HasField(field_name):
          getattr(self, field_name)._from_proto(getattr(proto, field_name))  # pylint: disable=protected-access

      # Process Non-Message type
      else:
        if field_descriptor.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          setattr(self, field_name, getattr(proto, field_name)[:])
        elif proto.HasField(field_name):
          setattr(self, field_name, getattr(proto, field_name))

    return self

  def merge_from_proto(self, proto: message.Message) -> "BaseModelCardField":
    """Merges the contents of the model card proto into current object."""
    current = self.to_proto()
    current.MergeFrom(proto)
    self.clear()
    return self._from_proto(current)

  def copy_from_proto(self, proto: message.Message) -> "BaseModelCardField":
    """Copies the contents of the model card proto into current object."""
    self.clear()
    return self._from_proto(proto)

  def _from_json(
      self, json_dict: Dict[str, Any], field: "BaseModelCardField"
  ) -> "BaseModelCardField":
    """Parses a JSON dictionary into the current object."""
    for subfield_key, subfield_json_value in json_dict.items():
      if subfield_key.startswith(validation.SCHEMA_VERSION_STRING):
        continue
      elif not hasattr(field, subfield_key):
        raise ValueError(
            "BaseModelCardField %s has no such field named '%s.'" %
            (field, subfield_key)
        )
      elif isinstance(subfield_json_value, dict):
        subfield_value = self._from_json(
            subfield_json_value, getattr(field, subfield_key)
        )
      elif isinstance(subfield_json_value, list):
        subfield_value = []
        for item in subfield_json_value:
          if isinstance(item, dict):
            new_object = \
              field.__annotations__[subfield_key].__args__[0]()  # pytype: disable=attribute-error
            subfield_value.append(self._from_json(item, new_object))
          else:  # if primitive
            subfield_value.append(item)
      else:
        subfield_value = subfield_json_value
      setattr(field, subfield_key, subfield_value)
    return field

  def to_json(self) -> str:
    """Convert this class object to json."""
    return json_lib.dumps(self.to_dict(), indent=2)

  def to_dict(self) -> Dict[str, Any]:
    """Convert your model card to a python dictionary."""
    # ignore None properties recursively to allow missing values.
    ignore_none = lambda properties: {k: v for k, v in properties if v}
    return dataclasses.asdict(self, dict_factory=ignore_none)

  def clear(self):
    """Clear the subfields of this BaseModelCardField."""
    for field_name, field_value in self.__dict__.items():
      if isinstance(field_value, BaseModelCardField):
        field_value.clear()
      elif isinstance(field_value, list):
        setattr(self, field_name, [])
      else:
        setattr(self, field_name, None)

  @classmethod
  def _get_type(cls, obj: Any):
    return type(obj)
