from anchor_generator import anchor_generator
#from Project.anchor_generators import multiple_grid_anchor_generator
#from Project.anchor_generators import multiscale_grid_anchor_generator
from protos import anchor_generator_pb2


def build(anchor_generator_config):
  """Builds an anchor generator based on the config.
  Args:
    anchor_generator_config: An anchor_generator.proto object containing the
      config for the desired anchor generator.
  Returns:
    Anchor generator based on the config.
  Raises:
    ValueError: On empty anchor generator proto.
  """
  if not isinstance(anchor_generator_config,
                    anchor_generator_pb2.AnchorGenerator):
    raise ValueError('anchor_generator_config not of type '
                     'anchor_generator_pb2.AnchorGenerator')
  if anchor_generator_config.WhichOneof(
      'anchor_generator_oneof') == 'grid_anchor_generator':
    grid_anchor_generator_config = anchor_generator_config.grid_anchor_generator
    return anchor_generator.GridAnchorGenerator(
        scales=[float(scale) for scale in grid_anchor_generator_config.scales],
        aspect_ratios=[float(aspect_ratio)
                       for aspect_ratio
                       in grid_anchor_generator_config.aspect_ratios],
        base_anchor_size=[grid_anchor_generator_config.height,
                          grid_anchor_generator_config.width],
        anchor_stride=[grid_anchor_generator_config.height_stride,
                       grid_anchor_generator_config.width_stride],
        anchor_offset=[grid_anchor_generator_config.height_offset,
                       grid_anchor_generator_config.width_offset])
  else:
    raise ValueError('Empty anchor generator.')
