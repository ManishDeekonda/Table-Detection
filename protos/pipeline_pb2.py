# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Project/protos/pipeline.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from protos import eval_pb2 as Project_dot_protos_dot_eval__pb2
from protos import graph_rewriter_pb2 as Project_dot_protos_dot_graph__rewriter__pb2
from protos import input_reader_pb2 as Project_dot_protos_dot_input__reader__pb2
from protos import model_pb2 as Project_dot_protos_dot_model__pb2
from protos import train_pb2 as Project_dot_protos_dot_train__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='Project/protos/pipeline.proto',
  package='Project.protos',
  syntax='proto2',
  serialized_pb=_b('\n\x1dProject/protos/pipeline.proto\x12\x0eProject.protos\x1a\x19Project/protos/eval.proto\x1a#Project/protos/graph_rewriter.proto\x1a!Project/protos/input_reader.proto\x1a\x1aProject/protos/model.proto\x1a\x1aProject/protos/train.proto\"\xdf\x02\n\x17TrainEvalPipelineConfig\x12-\n\x05model\x18\x01 \x01(\x0b\x32\x1e.Project.protos.DetectionModel\x12\x31\n\x0ctrain_config\x18\x02 \x01(\x0b\x32\x1b.Project.protos.TrainConfig\x12\x37\n\x12train_input_reader\x18\x03 \x01(\x0b\x32\x1b.Project.protos.InputReader\x12/\n\x0b\x65val_config\x18\x04 \x01(\x0b\x32\x1a.Project.protos.EvalConfig\x12\x36\n\x11\x65val_input_reader\x18\x05 \x01(\x0b\x32\x1b.Project.protos.InputReader\x12\x35\n\x0egraph_rewriter\x18\x06 \x01(\x0b\x32\x1d.Project.protos.GraphRewriter*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02')
  ,
  dependencies=[Project_dot_protos_dot_eval__pb2.DESCRIPTOR,Project_dot_protos_dot_graph__rewriter__pb2.DESCRIPTOR,Project_dot_protos_dot_input__reader__pb2.DESCRIPTOR,Project_dot_protos_dot_model__pb2.DESCRIPTOR,Project_dot_protos_dot_train__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_TRAINEVALPIPELINECONFIG = _descriptor.Descriptor(
  name='TrainEvalPipelineConfig',
  full_name='Project.protos.TrainEvalPipelineConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='Project.protos.TrainEvalPipelineConfig.model', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='train_config', full_name='Project.protos.TrainEvalPipelineConfig.train_config', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='train_input_reader', full_name='Project.protos.TrainEvalPipelineConfig.train_input_reader', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_config', full_name='Project.protos.TrainEvalPipelineConfig.eval_config', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eval_input_reader', full_name='Project.protos.TrainEvalPipelineConfig.eval_input_reader', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='graph_rewriter', full_name='Project.protos.TrainEvalPipelineConfig.graph_rewriter', index=5,
      number=6, type=11, cpp_type=10, label=1,
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
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(1000, 536870912), ],
  oneofs=[
  ],
  serialized_start=205,
  serialized_end=556,
)

_TRAINEVALPIPELINECONFIG.fields_by_name['model'].message_type = Project_dot_protos_dot_model__pb2._DETECTIONMODEL
_TRAINEVALPIPELINECONFIG.fields_by_name['train_config'].message_type = Project_dot_protos_dot_train__pb2._TRAINCONFIG
_TRAINEVALPIPELINECONFIG.fields_by_name['train_input_reader'].message_type = Project_dot_protos_dot_input__reader__pb2._INPUTREADER
_TRAINEVALPIPELINECONFIG.fields_by_name['eval_config'].message_type = Project_dot_protos_dot_eval__pb2._EVALCONFIG
_TRAINEVALPIPELINECONFIG.fields_by_name['eval_input_reader'].message_type = Project_dot_protos_dot_input__reader__pb2._INPUTREADER
_TRAINEVALPIPELINECONFIG.fields_by_name['graph_rewriter'].message_type = Project_dot_protos_dot_graph__rewriter__pb2._GRAPHREWRITER
DESCRIPTOR.message_types_by_name['TrainEvalPipelineConfig'] = _TRAINEVALPIPELINECONFIG

TrainEvalPipelineConfig = _reflection.GeneratedProtocolMessageType('TrainEvalPipelineConfig', (_message.Message,), dict(
  DESCRIPTOR = _TRAINEVALPIPELINECONFIG,
  __module__ = 'Project.protos.pipeline_pb2'
  # @@protoc_insertion_point(class_scope:Project.protos.TrainEvalPipelineConfig)
  ))
_sym_db.RegisterMessage(TrainEvalPipelineConfig)


# @@protoc_insertion_point(module_scope)
