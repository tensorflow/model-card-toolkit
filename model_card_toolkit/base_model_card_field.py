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
import json as json_lib
from typing import Any, Dict, List, Text, Union
import dataclasses

from google.protobuf import descriptor
from google.protobuf import message

MessageFieldType = Union["BaseModelCardField", Dict[Text, Any]]
PrimitiveFieldType = Union[float, int, Text]


class BaseModelCardField(abc.ABC):
  """Model card field base class.

  This is an abstract class. All the model card fields should inherit this class
  and override the _proto_type property to the corresponding proto type. This
  abstract class provides methods `copy_from_proto`, `merge_from_proto` and
  `to_proto` to convert the class from and to proto. The child class does not
  need to override this unless it needs some special process.
  """

  @property
  @abc.abstractmethod
  def _proto_type(self):
    """The proto type. Child class should overwrite this."""

  def is_empty(self) -> bool:
    for field_val in self.__dict__.values():
      if isinstance(field_val, BaseModelCardField):
        return field_val.is_empty()
      elif field_val:
        return False
    return True

  def to_proto(self) -> message.Message:
    """Convert this class object to the proto."""

    def _is_empty_field(field_val: Any):
      return not field_val or (isinstance(field_val, BaseModelCardField) and
                               field_val.is_empty())

    def _field_to_proto(field_val: MessageFieldType,
                        field_name: Text) -> message.Message:
      # Fields can be BaseModelCardField or Dict.
      # Only BaseModelCardField can be converted to Proto.
      # So if field is Dict, we need to convert it to BaseModelCardField first.
      if isinstance(field_val, dict):
        # Get field's class
        if isinstance(getattr(self, field_name), list):
          field_class = getattr(self, "__annotations__")[field_name].__args__[0]
        else:
          field_class = getattr(self, "__annotations__")[field_name]
        # Read field_val from dict
        field_val = field_class()._from_dict(  # pylint: disable=protected-access
            field_val)
      # Convert BaseModelCardField to Proto.
      return field_val.to_proto()

    def _populate_message_field(proto: message.Message, field_name: Text,
                                field_val: Union[MessageFieldType,
                                                 List[MessageFieldType]],
                                field_desc: descriptor.FieldDescriptor):
      if field_desc.label == descriptor.FieldDescriptor.LABEL_REPEATED:
        for nested_message in field_val:
          nested_message_proto = _field_to_proto(nested_message, field_name)
          getattr(proto, field_name).add().CopyFrom(nested_message_proto)  # pylint: disable=protected-access
      else:
        field_val_proto = _field_to_proto(field_val, field_name)
        getattr(proto, field_name).CopyFrom(field_val_proto)  # pylint: disable=protected-access

    def _populate_primitive_field(proto: message.Message, field_name: Text,
                                  field_val: Union[PrimitiveFieldType,
                                                   List[PrimitiveFieldType]],
                                  field_desc: descriptor.FieldDescriptor):
      if field_desc.label == descriptor.FieldDescriptor.LABEL_REPEATED:
        getattr(proto, field_name).extend(field_val)
      else:
        setattr(proto, field_name, str(field_val))

    proto = self._proto_type()
    for field_name, field_val in self.__dict__.items():
      if not hasattr(proto, field_name):
        raise ValueError("%s has no such field named '%s'." %
                         (type(proto), field_name))
      if _is_empty_field(field_val):
        continue
      field_desc = proto.DESCRIPTOR.fields_by_name[field_name]
      if field_desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
        _populate_message_field(proto, field_name, field_val, field_desc)
      else:
        _populate_primitive_field(proto, field_name, field_val, field_desc)
    return proto

  def _from_proto(self, proto: message.Message) -> "BaseModelCardField":
    """Convert proto to this class object."""
    if not isinstance(proto, self._proto_type):
      raise TypeError("%s is expected. However %s is provided." %
                      (self._proto_type, type(proto)))

    for field_desc in proto.DESCRIPTOR.fields:
      field_name = field_desc.name
      if not hasattr(self, field_name):
        raise ValueError("%s has no such field named '%s.'" %
                         (self, field_name))

      # Process Message type.
      if field_desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE:
        if field_desc.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          # Clean the list first.
          setattr(self, field_name, [])
          for p in getattr(proto, field_name):
            # To get the type hint of a list is not easy.
            field = self.__annotations__[field_name].__args__[0]()  # pytype: disable=attribute-error
            field._from_proto(p)  # pylint: disable=protected-access
            getattr(self, field_name).append(field)

        elif proto.HasField(field_name):
          getattr(self, field_name)._from_proto(getattr(proto, field_name))  # pylint: disable=protected-access

      # Process Non-Message type
      else:
        if field_desc.label == descriptor.FieldDescriptor.LABEL_REPEATED:
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

  def to_json(self) -> Text:
    """Convert this class object to json."""
    return json_lib.dumps(self.to_dict(), indent=2)

  def to_dict(self) -> Dict[Text, Any]:
    """Convert your model card to a python dictionary."""
    # ignore None properties recursively to allow missing values.
    ignore_none = lambda properties: {k: v for k, v in properties if v}
    return dataclasses.asdict(self, dict_factory=ignore_none)

  def _from_dict(self, dict_message: Dict[Text, Any]) -> "BaseModelCardField":
    """Convert dict to this class object."""

    for field_name, field_val in dict_message.items():
      if not hasattr(self, field_name):
        raise ValueError("%s has no such field named '%s.'" %
                         (self, field_name))

      if isinstance(field_val, dict):
        setattr(self, field_name,
                getattr(self, field_name)._from_dict(field_val))  # pylint: disable=protected-access
      else:
        setattr(self, field_name, field_val)

    return self

  def clear(self):
    """Clear the subfields of this BaseModelCardField."""
    for field_name, field_val in self.__dict__.items():
      if isinstance(field_val, BaseModelCardField):
        field_val.clear()
      elif isinstance(field_val, list):
        setattr(self, field_name, [])
      else:
        setattr(self, field_name, None)

  @classmethod
  def _get_type(cls, obj: Any):
    return type(obj)
