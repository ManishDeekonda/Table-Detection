# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Project/protos/model.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from protos import faster_rcnn_pb2 as Project_dot_protos_dot_faster__rcnn__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='Project/protos/model.proto',
  package='Project.protos',
  syntax='proto2',
  serialized_pb=_b('\n\x1aProject/protos/model.proto\x12\x0eProject.protos\x1a Project/protos/faster_rcnn.proto\"L\n\x0e\x44\x65tectionModel\x12\x31\n\x0b\x66\x61ster_rcnn\x18\x01 \x01(\x0b\x32\x1a.Project.protos.FasterRcnnH\x00\x42\x07\n\x05model')
  ,
  dependencies=[Project_dot_protos_dot_faster__rcnn__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_DETECTIONMODEL = _descriptor.Descriptor(
  name='DetectionModel',
  full_name='Project.protos.DetectionModel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='faster_rcnn', full_name='Project.protos.DetectionModel.faster_rcnn', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='model', full_name='Project.protos.DetectionModel.model',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=80,
  serialized_end=156,
)

_DETECTIONMODEL.fields_by_name['faster_rcnn'].message_type = Project_dot_protos_dot_faster__rcnn__pb2._FASTERRCNN
_DETECTIONMODEL.oneofs_by_name['model'].fields.append(
  _DETECTIONMODEL.fields_by_name['faster_rcnn'])
_DETECTIONMODEL.fields_by_name['faster_rcnn'].containing_oneof = _DETECTIONMODEL.oneofs_by_name['model']
DESCRIPTOR.message_types_by_name['DetectionModel'] = _DETECTIONMODEL

DetectionModel = _reflection.GeneratedProtocolMessageType('DetectionModel', (_message.Message,), dict(
  DESCRIPTOR = _DETECTIONMODEL,
  __module__ = 'Project.protos.model_pb2'
  # @@protoc_insertion_point(class_scope:Project.protos.DetectionModel)
  ))
_sym_db.RegisterMessage(DetectionModel)


# @@protoc_insertion_point(module_scope)