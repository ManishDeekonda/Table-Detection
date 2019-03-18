from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class IdentityContextManager(object):
  """Returns an identity context manager that does nothing.
  This is helpful in setting up conditional `with` statement as below:
  with slim.arg_scope(x) if use_slim_scope else IdentityContextManager():
    do_stuff()
  """

  def __enter__(self):
    return None

  def __exit__(self, exec_type, exec_value, traceback):
    del exec_type
    del exec_value
    del traceback
    return False

