import math
import numpy as np
import six

import tensorflow as tf
from intermediate import shape_utils
from intermediate import box_list_ops
from anchor_generator import boxlist as box_list

def expanded_shape(orig_shape, start_dim, num_dims):
  """Inserts multiple ones into a shape vector.
  Inserts an all-1 vector of length num_dims at position start_dim into a shape.
  Can be combined with tf.reshape to generalize tf.expand_dims.
  Args:
    orig_shape: the shape into which the all-1 vector is added (int32 vector)
    start_dim: insertion position (int scalar)
    num_dims: length of the inserted all-1 vector (int scalar)
  Returns:
    An int32 vector of length tf.size(orig_shape) + num_dims.
  """
  with tf.name_scope('ExpandedShape'):
    start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
    before = tf.slice(orig_shape, [0], start_dim)
    add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
    after = tf.slice(orig_shape, start_dim, [-1])
    new_shape = tf.concat([before, add_shape, after], 0)
    return new_shape

def normalized_to_image_coordinates(normalized_boxes, image_shape,
                                    parallel_iterations=32):
  """Converts a batch of boxes from normal to image coordinates.
  Args:
    normalized_boxes: a float32 tensor of shape [None, num_boxes, 4] in
      normalized coordinates.
    image_shape: a float32 tensor of shape [4] containing the image shape.
    parallel_iterations: parallelism for the map_fn op.
  Returns:
    absolute_boxes: a float32 tensor of shape [None, num_boxes, 4] containg the
      boxes in image coordinates.
  """
  def _to_absolute_coordinates(normalized_boxes):
    return box_list_ops.to_absolute_coordinates(
        box_list.BoxList(normalized_boxes),
        image_shape[1], image_shape[2], check_range=False).get()

  absolute_boxes = shape_utils.static_or_dynamic_map_fn(
      _to_absolute_coordinates,
      elems=(normalized_boxes),
      dtype=tf.float32,
      parallel_iterations=parallel_iterations,
      back_prop=True)
  return absolute_boxes

def meshgrid(x, y):
  """Tiles the contents of x and y into a pair of grids.
  Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
  are vectors. Generally, this will give:
  xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
  ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)
  Keep in mind that the order of the arguments and outputs is reverse relative
  to the order of the indices they go into, done for compatibility with numpy.
  The output tensors have the same shapes.  Specifically:
  xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
  ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())
  Args:
    x: A tensor of arbitrary shape and rank. xgrid will contain these values
       varying in its last dimensions.
    y: A tensor of arbitrary shape and rank. ygrid will contain these values
       varying in its first dimensions.
  Returns:
    A tuple of tensors (xgrid, ygrid).
  """
  with tf.name_scope('Meshgrid'):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
    y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

    xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
    new_shape = y.get_shape().concatenate(x.get_shape())
    xgrid.set_shape(new_shape)
    ygrid.set_shape(new_shape)

    return xgrid, ygrid
def padded_one_hot_encoding(indices, depth, left_pad):
  """Returns a zero padded one-hot tensor.
  This function converts a sparse representation of indices (e.g., [4]) to a
  zero padded one-hot representation (e.g., [0, 0, 0, 0, 1] with depth = 4 and
  left_pad = 1). If `indices` is empty, the result will simply be a tensor of
  shape (0, depth + left_pad). If depth = 0, then this function just returns
  `None`.
  Args:
    indices: an integer tensor of shape [num_indices].
    depth: depth for the one-hot tensor (integer).
    left_pad: number of zeros to left pad the one-hot tensor with (integer).
  Returns:
    padded_onehot: a tensor with shape (num_indices, depth + left_pad). Returns
      `None` if the depth is zero.
  Raises:
    ValueError: if `indices` does not have rank 1 or if `left_pad` or `depth are
      either negative or non-integers.
  TODO(rathodv): add runtime checks for depth and indices.
  """
  if depth < 0 or not isinstance(depth, six.integer_types):
    raise ValueError('`depth` must be a non-negative integer.')
  if left_pad < 0 or not isinstance(left_pad, six.integer_types):
    raise ValueError('`left_pad` must be a non-negative integer.')
  if depth == 0:
    return None

  rank = len(indices.get_shape().as_list())
  if rank != 1:
    raise ValueError('`indices` must have rank 1, but has rank=%s' % rank)

  def one_hot_and_pad():
    one_hot = tf.cast(tf.one_hot(tf.cast(indices, tf.int64), depth,
                                 on_value=1, off_value=0), tf.float32)
    return tf.pad(one_hot, [[0, 0], [left_pad, 0]], mode='CONSTANT')
  result = tf.cond(tf.greater(tf.size(indices), 0), one_hot_and_pad,
                   lambda: tf.zeros((depth + left_pad, 0)))
  return tf.reshape(result, [-1, depth + left_pad])
def merge_boxes_with_multiple_labels(boxes, classes, num_classes):
  """Merges boxes with same coordinates and returns K-hot encoded classes.
  Args:
    boxes: A tf.float32 tensor with shape [N, 4] holding N boxes.
    classes: A tf.int32 tensor with shape [N] holding class indices.
      The class index starts at 0.
    num_classes: total number of classes to use for K-hot encoding.
  Returns:
    merged_boxes: A tf.float32 tensor with shape [N', 4] holding boxes,
      where N' <= N.
    class_encodings: A tf.int32 tensor with shape [N', num_classes] holding
      k-hot encodings for the merged boxes.
    merged_box_indices: A tf.int32 tensor with shape [N'] holding original
      indices of the boxes.
  """
  def merge_numpy_boxes(boxes, classes, num_classes):
    """Python function to merge numpy boxes."""
    if boxes.size < 1:
      return (np.zeros([0, 4], dtype=np.float32),
              np.zeros([0, num_classes], dtype=np.int32),
              np.zeros([0], dtype=np.int32))
    box_to_class_indices = {}
    for box_index in range(boxes.shape[0]):
      box = tuple(boxes[box_index, :].tolist())
      class_index = classes[box_index]
      if box not in box_to_class_indices:
        box_to_class_indices[box] = [box_index, np.zeros([num_classes])]
      box_to_class_indices[box][1][class_index] = 1
    merged_boxes = np.vstack(box_to_class_indices.keys()).astype(np.float32)
    class_encodings = [item[1] for item in box_to_class_indices.values()]
    class_encodings = np.vstack(class_encodings).astype(np.int32)
    merged_box_indices = [item[0] for item in box_to_class_indices.values()]
    merged_box_indices = np.array(merged_box_indices).astype(np.int32)
    return merged_boxes, class_encodings, merged_box_indices

  merged_boxes, class_encodings, merged_box_indices = tf.py_func(
      merge_numpy_boxes, [boxes, classes, num_classes],
      [tf.float32, tf.int32, tf.int32])
  merged_boxes = tf.reshape(merged_boxes, [-1, 4])
  class_encodings = tf.reshape(class_encodings, [-1, num_classes])
  merged_box_indices = tf.reshape(merged_box_indices, [-1])
  return merged_boxes, class_encodings, merged_box_indices

def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.
  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])
  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.
  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  size = tf.to_int32(size)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value

  return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
                           [zeros, values])
def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):
  """Creates dense vector with indices set to specific value and rest to zeros.
  This function exists because it is unclear if it is safe to use
    tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
  with indices which are not ordered.
  This function accepts a dynamic size (e.g. tf.shape(tensor)[0])
  Args:
    indices: 1d Tensor with integer indices which are to be set to
        indices_values.
    size: scalar with size (integer) of output Tensor.
    indices_value: values of elements specified by indices in the output vector
    default_value: values of other elements in the output vector.
    dtype: data type.
  Returns:
    dense 1D Tensor of shape [size] with indices set to indices_values and the
        rest set to default_value.
  """
  size = tf.to_int32(size)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value

  return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
                           [zeros, values])

def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
  """Transforms the box masks back to full image masks.
  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.
  Args:
    box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.
  Returns:
    A tf.float32 tensor of size [num_masks, image_height, image_width].
  """
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    return tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_ind=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        extrapolation_value=0.0)
  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))
  return tf.squeeze(image_masks, axis=3)



def reduce_sum_trailing_dimensions(tensor, ndims):
  """Computes sum across all dimensions following first `ndims` dimensions."""
  return tf.reduce_sum(tensor, axis=tuple(range(ndims, tensor.shape.ndims)))

def matmul_gather_on_zeroth_axis(params, indices, scope=None):
  """Matrix multiplication based implementation of tf.gather on zeroth axis.
  TODO(rathodv, jonathanhuang): enable sparse matmul option.
  Args:
    params: A float32 Tensor. The tensor from which to gather values.
      Must be at least rank 1.
    indices: A Tensor. Must be one of the following types: int32, int64.
      Must be in range [0, params.shape[0])
    scope: A name for the operation (optional).
  Returns:
    A Tensor. Has the same type as params. Values from params gathered
    from indices given by indices, with shape indices.shape + params.shape[1:].
  """
  with tf.name_scope(scope, 'MatMulGather'):
    params_shape = shape_utils.combined_static_and_dynamic_shape(params)
    indices_shape = shape_utils.combined_static_and_dynamic_shape(indices)
    params2d = tf.reshape(params, [params_shape[0], -1])
    indicator_matrix = tf.one_hot(indices, params_shape[0])
    gathered_result_flattened = tf.matmul(indicator_matrix, params2d)
    return tf.reshape(gathered_result_flattened,
                      tf.stack(indices_shape + params_shape[1:]))
