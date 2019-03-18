import functools
import tensorflow as tf

from intermediate import standard_fields as fields
from train_file import tf_example_decoder
from protos import input_reader_pb2
from train_file import dataset_util

def make_initializable_iterator(dataset):
  """Creates an iterator, and initializes tables.
  This is useful in cases where make_one_shot_iterator wouldn't work because
  the graph contains a hash table that needs to be initialized.
  Args:
    dataset: A `tf.data.Dataset` object.
  Returns:
    A `tf.data.Iterator`.
  """
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return iterator


def _get_padding_shapes(dataset, max_num_boxes=None, num_classes=None,
                        spatial_image_shape=None):
  """Returns shapes to pad dataset tensors to before batching.
  Args:
    dataset: tf.data.Dataset object.
    max_num_boxes: Max number of groundtruth boxes needed to computes shapes for
      padding.
    num_classes: Number of classes in the dataset needed to compute shapes for
      padding.
    spatial_image_shape: A list of two integers of the form [height, width]
      containing expected spatial shape of the image.
  Returns:
    A dictionary keyed by fields.InputDataFields containing padding shapes for
    tensors in the dataset.
  Raises:
    ValueError: If groundtruth classes is neither rank 1 nor rank 2.
  """

  if not spatial_image_shape or spatial_image_shape == [-1, -1]:
    height, width = None, None
  else:
    height, width = spatial_image_shape  # pylint: disable=unpacking-non-sequence

  num_additional_channels = 0
  if fields.InputDataFields.image_additional_channels in dataset.output_shapes:
    num_additional_channels = dataset.output_shapes[
        fields.InputDataFields.image_additional_channels].dims[2].value
  padding_shapes = {
      # Additional channels are merged before batching.
      fields.InputDataFields.image: [
          height, width, 3 + num_additional_channels
      ],
      fields.InputDataFields.image_additional_channels: [
          height, width, num_additional_channels
      ],
      fields.InputDataFields.source_id: [],
      fields.InputDataFields.filename: [],
      fields.InputDataFields.key: [],
      fields.InputDataFields.groundtruth_difficult: [max_num_boxes],
      fields.InputDataFields.groundtruth_boxes: [max_num_boxes, 4],
      fields.InputDataFields.groundtruth_instance_masks: [
          max_num_boxes, height, width
      ],
      fields.InputDataFields.groundtruth_is_crowd: [max_num_boxes],
      fields.InputDataFields.groundtruth_group_of: [max_num_boxes],
      fields.InputDataFields.groundtruth_area: [max_num_boxes],
      fields.InputDataFields.groundtruth_weights: [max_num_boxes],
      fields.InputDataFields.num_groundtruth_boxes: [],
      fields.InputDataFields.groundtruth_label_types: [max_num_boxes],
      fields.InputDataFields.groundtruth_label_scores: [max_num_boxes],
      fields.InputDataFields.true_image_shape: [3],
      fields.InputDataFields.multiclass_scores: [
          max_num_boxes, num_classes + 1 if num_classes is not None else None
      ],
  }
  # Determine whether groundtruth_classes are integers or one-hot encodings, and
  # apply batching appropriately.
  classes_shape = dataset.output_shapes[
      fields.InputDataFields.groundtruth_classes]
  if len(classes_shape) == 1:  # Class integers.
    padding_shapes[fields.InputDataFields.groundtruth_classes] = [max_num_boxes]
  elif len(classes_shape) == 2:  # One-hot or k-hot encoding.
    padding_shapes[fields.InputDataFields.groundtruth_classes] = [
        max_num_boxes, num_classes]
  else:
    raise ValueError('Groundtruth classes must be a rank 1 tensor (classes) or '
                     'rank 2 tensor (one-hot encodings)')

  if fields.InputDataFields.original_image in dataset.output_shapes:
    padding_shapes[fields.InputDataFields.original_image] = [
        None, None, 3 + num_additional_channels
    ]
  if fields.InputDataFields.groundtruth_keypoints in dataset.output_shapes:
    tensor_shape = dataset.output_shapes[fields.InputDataFields.
                                         groundtruth_keypoints]
    padding_shape = [max_num_boxes, tensor_shape[1].value,
                     tensor_shape[2].value]
    padding_shapes[fields.InputDataFields.groundtruth_keypoints] = padding_shape
  if (fields.InputDataFields.groundtruth_keypoint_visibilities
      in dataset.output_shapes):
    tensor_shape = dataset.output_shapes[fields.InputDataFields.
                                         groundtruth_keypoint_visibilities]
    padding_shape = [max_num_boxes, tensor_shape[1].value]
    padding_shapes[fields.InputDataFields.
                   groundtruth_keypoint_visibilities] = padding_shape
  return {tensor_key: padding_shapes[tensor_key]
          for tensor_key, _ in dataset.output_shapes.items()}


def build(input_reader_config,
          transform_input_data_fn=None,
          batch_size=None,
          max_num_boxes=None,
          num_classes=None,
          spatial_image_shape=None,
          num_additional_channels=0):
  """Builds a tf.data.Dataset.
  Builds a tf.data.Dataset by applying the `transform_input_data_fn` on all
  records. Applies a padded batch to the resulting dataset.
  Args:
    input_reader_config: A input_reader_pb2.InputReader object.
    transform_input_data_fn: Function to apply to all records, or None if
      no extra decoding is required.
    batch_size: Batch size. If None, batching is not performed.
    max_num_boxes: Max number of groundtruth boxes needed to compute shapes for
      padding. If None, will use a dynamic shape.
    num_classes: Number of classes in the dataset needed to compute shapes for
      padding. If None, will use a dynamic shape.
    spatial_image_shape: A list of two integers of the form [height, width]
      containing expected spatial shape of the image after applying
      transform_input_data_fn. If None, will use dynamic shapes.
    num_additional_channels: Number of additional channels to use in the input.
  Returns:
    A tf.data.Dataset based on the input_reader_config.
  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not config.input_path:
      raise ValueError('At least one input path must be specified in '
                       '`input_reader_config`.')

    label_map_proto_file = None
    if input_reader_config.HasField('label_map_path'):
      label_map_proto_file = input_reader_config.label_map_path
    decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=input_reader_config.load_instance_masks,
        instance_mask_type=input_reader_config.mask_type,
        label_map_proto_file=label_map_proto_file,
        use_display_name=input_reader_config.use_display_name,
        num_additional_channels=num_additional_channels)

    def process_fn(value):
      processed = decoder.decode(value)
      if transform_input_data_fn is not None:
        return transform_input_data_fn(processed)
      return processed

    dataset = dataset_util.read_dataset(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
        process_fn, config.input_path[:], input_reader_config)

    if batch_size:
      padding_shapes = _get_padding_shapes(dataset, max_num_boxes, num_classes,
                                           spatial_image_shape)
      dataset = dataset.apply(
          tf.contrib.data.padded_batch_and_drop_remainder(batch_size,
                                                          padding_shapes))
    return dataset

  raise ValueError('Unsupported input_reader_config.')
