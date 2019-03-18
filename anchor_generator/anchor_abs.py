from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


class AnchorGenerator(object):
  """Abstract base class for anchor generators."""
  __metaclass__ = ABCMeta

  @abstractmethod
  def name_scope(self):
    """Name scope.
    Must be defined by implementations.
    Returns:
      a string representing the name scope of the anchor generation operation.
    """
    pass

  @property
  def check_num_anchors(self):
    """Whether to dynamically check the number of anchors generated.
    Can be overridden by implementations that would like to disable this
    behavior.
    Returns:
      a boolean controlling whether the Generate function should dynamically
      check the number of anchors generated against the mathematically
      expected number of anchors.
    """
    return True

  @abstractmethod
  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.
    Returns:
      a list of integers, one for each expected feature map to be passed to
      the `generate` function.
    """
    pass

  def generate(self, feature_map_shape_list, **params):
    """Generates a collection of bounding boxes to be used as anchors.
    TODO(rathodv): remove **params from argument list and make stride and
      offsets (for multiple_grid_anchor_generator) constructor arguments.
    Args:
      feature_map_shape_list: list of (height, width) pairs in the format
        [(height_0, width_0), (height_1, width_1), ...] that the generated
        anchors must align with.  Pairs can be provided as 1-dimensional
        integer tensors of length 2 or simply as tuples of integers.
      **params: parameters for anchor generation op
    Returns:
      boxes_list: a list of BoxLists each holding anchor boxes corresponding to
        the input feature map shapes.
    Raises:
      ValueError: if the number of feature map shapes does not match the length
        of NumAnchorsPerLocation.
    """
    if self.check_num_anchors and (
        len(feature_map_shape_list) != len(self.num_anchors_per_location())):
      raise ValueError('Number of feature maps is expected to equal the length '
                       'of `num_anchors_per_location`.')
    with tf.name_scope(self.name_scope()):
      anchors_list = self._generate(feature_map_shape_list, **params)
      if self.check_num_anchors:
        with tf.control_dependencies([
            self._assert_correct_number_of_anchors(
                anchors_list, feature_map_shape_list)]):
          for item in anchors_list:
            item.set(tf.identity(item.get()))
      return anchors_list

  @abstractmethod
  def _generate(self, feature_map_shape_list, **params):
    """To be overridden by implementations.
    Args:
      feature_map_shape_list: list of (height, width) pairs in the format
        [(height_0, width_0), (height_1, width_1), ...] that the generated
        anchors must align with.
      **params: parameters for anchor generation op
    Returns:
      boxes_list: a list of BoxList, each holding a collection of N anchor
        boxes.
    """
    pass

  def _assert_correct_number_of_anchors(self, anchors_list,
                                        feature_map_shape_list):
    """Assert that correct number of anchors was generated.
    Args:
      anchors_list: A list of box_list.BoxList object holding anchors generated.
      feature_map_shape_list: list of (height, width) pairs in the format
        [(height_0, width_0), (height_1, width_1), ...] that the generated
        anchors must align with.
    Returns:
      Op that raises InvalidArgumentError if the number of anchors does not
        match the number of expected anchors.
    """
    expected_num_anchors = 0
    actual_num_anchors = 0
    for num_anchors_per_location, feature_map_shape, anchors in zip(
        self.num_anchors_per_location(), feature_map_shape_list, anchors_list):
      expected_num_anchors += (num_anchors_per_location
                               * feature_map_shape[0]
                               * feature_map_shape[1])
      actual_num_anchors += anchors.num_boxes()
    return tf.assert_equal(expected_num_anchors, actual_num_anchors)

