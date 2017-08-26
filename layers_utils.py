from __future__ import print_function

import numpy as np
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


def clear_layers_name():
    """Clear all layer names in set_keep['_layers_name_list'].

    Enable layer name reuse.
    """
    set_keep['_layers_name_list'] = []


def flatten_reshape(variable, name=''):
    """Reshapes high-dimension input to a vector.

    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    variable : a tensorflow variable
    name : a string or None
        An optional name to attach to this layer.

    """
    dim = 1
    for d in variable.get_shape()[1:].as_list():
        dim *= d

    return tf.reshape(variable, shape=[-1, dim], name=name)


def initialize_global_variables(sess=None):
    """Excute ``sess.run(tf.global_variables_initializer())``.

    Parameters
    ----------
    sess : a Session

    """
    assert sess is not None
    sess.run(tf.global_variables_initializer())


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

        if (name in set_keep['_layers_name_list']) and set_keep['name_reuse'] == False:
            raise Exception("Layer '%s' already exists, please choice other 'name' or reuse this layer\
            \nHint : Use different name for different 'Layer' (The name is used to control parameter sharing)" % name)
        else:
            self.name = name
            if name not in ['', None, False]:
                set_keep['_layers_name_list'].append(name)

    def print_params(self, details=True):
        """Print all info of parameters in the network."""
        for i, p in enumerate(self.all_params):
            if details:
                try:
                    val = p.eval()
                    print("  param {:3}: {:20} {:15}    {} (mean: {:<18}, median: {:<18}, std: {:<18})   ".format(
                        i, p.name, str(val.shape), p.dtype.name, val.mean(), np.median(val), val.std()))
                except Exception as e:
                    print(str(e))
                    raise Exception(
                        "Hint: print params details after tl.layers.initialize_global_variables(sess) or use network.print_params(False).")

            else:
                print("  param {:3}: {:20} {:15}    {}".format(i, p.name, str(p.get_shape()), p.dtype.name))
        print("  num of params: %d" % self.count_params())

    def print_layers(self):
        """Print all info of layers in the network."""
        for i, layer in enumerate(self.all_layers):
            print("  layer {:3}: {:20} {:15}    {}".format(i, layer.name, str(layer.get_shape()), layer.dtype.name))

    def count_params(self):
        """Return the number of parameters in the network."""
        n_params = 0
        for i, p in enumerate(self.all_params):
            n = 1
            for s in p.get_shape():
                try:
                    s = int(s)
                except TypeError:
                    s = 1
                if s:
                    n = n * s
            n_params = n_params + n
        return n_params

    def __str__(self):
        """Return the class name."""
        return "  Last layer is: %s" % self.__class__.__name__


class InputLayer(Layer):
    """Construct the input layer.

    The :class:`InputLayer` class is the starting layer of a neural network.

    Parameters
    ----------
    inputs : a placeholder or tensor
        The input tensor data.
    name : a string or None
        An optional name to attach to this layer.

    """

    def __init__(self, inputs=None, name='input_layer'):
        Layer.__init__(self, inputs=inputs, name=name)
        print("  [TL] InputLayer  %s: %s" % (self.name, inputs.get_shape()))
        self.outputs = inputs
        self.all_layers = []
        self.all_params = []
        self.all_drop = {}


class DenseLayer(Layer):
    """Construct the dense layer.

    The :class:`DenseLayer` class is a fully connected layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    n_units : int
        The number of units of the layer.
    act : activation function
        The function that is applied to the layer activations.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable.
    b_init_args : dictionary
        The arguments for the biases tf.get_variable.
    name : a string or None
        An optional name to attach to this layer.

    """

    def __init__(self,
                 layer=None,
                 n_units=100,
                 act=tf.identity,
                 W_init=tf.truncated_normal_initializer(stddev=0.1),
                 b_init=tf.constant_initializer(value=0.0),
                 W_init_args={},
                 b_init_args={},
                 name='dense_layer'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        print("  [TL] DenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W',
                                shape=[n_in, n_units],
                                initializer=W_init,
                                **W_init_args)
            if b_init is not None:
                try:
                    b = tf.get_variable(name='b',
                                        shape=[n_units],
                                        initializer=b_init,
                                        **b_init_args)
                except:
                    b = tf.get_variable(name='b',
                                        initializer=b_init,
                                        **b_init_args)
                self.outputs = act(tf.matmul(self.inputs, W) + b)
            else:
                self.outputs = act(tf.matmul(self.inputs, W))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init is not None:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


class DropoutLayer(Layer):
    """Construct dropout layer.

    The :class:`DropoutLayer` class is a noise layer which randomly set some
    values to zero by a given keeping probability.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    keep : float
        The keeping probability, the lower more values will be set to zero.
    is_fix : boolean
        Default False, if True, the keeping probability is fixed and cannot be
        changed via feed_dict.
    is_train : boolean
        If False, skip this layer, default is True.
    seed : int or None
        An integer or None to create random seed.
    name : a string or None
        An optional name to attach to this layer.

    Notes
    -----
    - A frequent question regarding :class:`DropoutLayer` is that why it donot have
    `is_train` like :class:`BatchNormLayer`. In many simple cases, user may find it
    is better to use one inference instead of two inferences for training and testing
    seperately, :class:`DropoutLayer` allows you to control the dropout rate via
    `feed_dict`. However, you can fix the keeping probability by setting `is_fix` to True.

    """

    def __init__(self,
                 layer=None,
                 keep=0.5,
                 is_fix=False,
                 is_train=True,
                 seed=None,
                 name='dropout_layer'):
        Layer.__init__(self, name=name)
        if is_train is False:
            print("  [TL] skip DropoutLayer")
            self.outputs = layer.outputs
            self.all_layers = list(layer.all_layers)
            self.all_params = list(layer.all_params)
            self.all_drop = dict(layer.all_drop)
        else:
            self.inputs = layer.outputs
            print("  [TL] DropoutLayer %s: keep:%f is_fix:%s" % (self.name, keep, is_fix))

            if is_fix:
                self.outputs = tf.nn.dropout(self.inputs, keep, seed=seed, name=name)
            else:
                set_keep[name] = tf.placeholder(tf.float32)
                self.outputs = tf.nn.dropout(self.inputs, set_keep[name], seed=seed, name=name)

            self.all_layers = list(layer.all_layers)
            self.all_params = list(layer.all_params)
            self.all_drop = dict(layer.all_drop)
            if is_fix:
                self.all_drop.update({set_keep[name]: keep})
            self.all_layers.extend([self.outputs])


class Conv2dLayer(Layer):
    """Construct conv2d layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    shape : list of shape
        shape of the filters, [filter_height, filter_width, in_channels, out_channels].
    strides : a list of ints.
        The stride of the sliding window for each dimension of input.
        It Must be in the same order as the dimension specified with format.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable().
    b_init_args : dictionary
        The arguments for the biases tf.get_variable().
    use_cudnn_on_gpu : bool, default is None.
    data_format : string "NHWC" or "NCHW", default is "NHWC"
    name : a string or None
        An optional name to attach to this layer.

    Notes
    -----
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    """

    def __init__(self,
                 layer=None,
                 act=tf.identity,
                 shape=[5, 5, 1, 100],
                 strides=[1, 1, 1, 1],
                 padding='SAME',
                 W_init=tf.truncated_normal_initializer(stddev=0.02),
                 b_init=tf.constant_initializer(value=0),
                 W_init_args={},
                 b_init_args={},
                 use_cudnn_on_gpu=None,
                 data_format=None,
                 name='cnn_layer'):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] Conv2dLayer %s: shape:%s strides:%s pad:%s act:%s" %
              (self.name, str(shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W_conv2d',
                                shape=shape,
                                initializer=W_init,
                                **W_init_args)
            if b_init:
                b = tf.get_variable(name='b_conv2d',
                                    shape=[shape[-1]],
                                    initializer=b_init,
                                    **b_init_args)
                self.outputs = act(tf.nn.conv2d(input=self.inputs,
                                                filter=W,
                                                strides=strides,
                                                padding=padding,
                                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                                data_format=data_format) + b)
            else:
                self.outputs = act(tf.nn.conv2d(input=self.inputs,
                                                filter=W,
                                                strides=strides,
                                                padding=padding,
                                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                                data_format=data_format))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


if __name__ == '__main__':
    # basic = Layer()
    # basic.print_params()
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    x = InputLayer(inputs=x, name='input')
    x = DenseLayer(layer=x)
    x.print_params(False)
