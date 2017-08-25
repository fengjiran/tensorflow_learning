import tensorflow as tf

set_keep = globals()
set_keep['_layers_name_list'] = []
set_keep['name_reuse'] = False


def set_name_reuse(enable=True):
    """Enable or disable reuse layer name.

    By default, each layer must has unique name. When you want two or more
    input placeholder (inference) share the same model parameters, you need
    to enable layer name reuse, then allow the parameters have same name scope.

    Parameters
    ----------
    enable : boolean, enable name reuse. (None means False).

    """
    set_keep['name_reuse'] = enable


class Layer(object):
    """Construct the basic layer.

    The :class:`Layer` class represents a single layer of a neural network. It
    should be subclassed when implementing new types of layers.
    Because each layer can keep track of the layer(s) feeding into it, a
    network's output :class:`Layer` instance can double as a handle to the full
    network.

    Parameters
    ----------
    inputs : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    name : a string or None
        An optional name to attach to this layer.

    """

    def __init__(self, inputs=None, name='layer'):
        self.inputs = inputs
        scope_name = tf.get_variable_scope().name

        if scope_name:
            name = scope_name + '/' + name

        if (name in set_keep['_layers_name_list']) and name_reuse == False:
            raise Exception("Layer '%s' already exists, please choice other 'name' or reuse this layer\
            \nHint : Use different name for different 'Layer' (The name is used to control parameter sharing)" % name)
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)


if __name__ == '__main__':
    basic = Layer()
