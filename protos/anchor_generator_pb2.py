# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Project/protos/anchor_generator.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from protos import grid_anchor_generator_pb2 as Project_dot_protos_dot_grid__anchor__generator__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='Project/protos/anchor_generator.proto',
  package='Project.protos',
  syntax='proto2',
  serialized_pb=_b('\n%Project/protos/anchor_generator.proto\x12\x0eProject.protos\x1a*Project/protos/grid_anchor_generator.proto\"q\n\x0f\x41nchorGenerator\x12\x44\n\x15grid_anchor_generator\x18\x01 \x01(\x0b\x32#.Project.protos.GridAnchorGeneratorH\x00\x42\x18\n\x16\x61nchor_generator_oneof')
  ,
  dependencies=[Project_dot_protos_dot_grid__anchor__generator__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_ANCHORGENERATOR = _descriptor.Descriptor(
  name='AnchorGenerator',
  full_name='Project.protos.AnchorGenerator',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='grid_anchor_generator', full_name='Project.protos.AnchorGenerator.grid_anchor_generator', index=0,
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
      name='anchor_generator_oneof', full_name='Project.protos.AnchorGenerator.anchor_generator_oneof',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=101,
  serialized_end=214,
)

_ANCHORGENERATOR.fields_by_name['grid_anchor_generator'].message_type = Project_dot_protos_dot_grid__anchor__generator__pb2._GRIDANCHORGENERATOR
_ANCHORGENERATOR.oneofs_by_name['anchor_generator_oneof'].fields.append(
  _ANCHORGENERATOR.fields_by_name['grid_anchor_generator'])
_ANCHORGENERATOR.fields_by_name['grid_anchor_generator'].containing_oneof = _ANCHORGENERATOR.oneofs_by_name['anchor_generator_oneof']
DESCRIPTOR.message_types_by_name['AnchorGenerator'] = _ANCHORGENERATOR

AnchorGenerator = _reflection.GeneratedProtocolMessageType('AnchorGenerator', (_message.Message,), dict(
  DESCRIPTOR = _ANCHORGENERATOR,
  __module__ = 'Project.protos.anchor_generator_pb2'
  # @@protoc_insertion_point(class_scope:Project.protos.AnchorGenerator)
  ))
_sym_db.RegisterMessage(AnchorGenerator)


# @@protoc_insertion_point(module_scope)
