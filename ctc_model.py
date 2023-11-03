import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

def leaky_relu(features, alpha=0.2, name=None):
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)

# Returns default model parameters
def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = 16
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
    params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params

def ctc_crnn(params):
    # TODO Assert parameters
    input = tf.compat.v1.placeholder(shape=(None,
                                   params['img_height'],
                                   params['img_width'],
                                   params['img_channels']),  # [batch, height, width, channels]
                            dtype=tf.float32,
                            name='model_input')
    input_shape = tf.compat.v1.shape(input)

    # Convolutional blocks
    x = input
    width_reduction = 1
    height_reduction = 1
    for i in range(params['conv_blocks']):
        x = tf.compat.v1.layers.conv2d(
            inputs=x,
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same",
            activation=None)

        x = tf.compat.v1.layers.batch_normalization(x)
        x = leaky_relu(x)

        x = tf.compat.v1.layers.max_pooling2d(inputs=x,
                                    pool_size=params['conv_pooling_size'][i],
                                    strides=params['conv_pooling_size'][i])

        width_reduction = width_reduction * params['conv_pooling_size'][i][1]
        height_reduction = height_reduction * params['conv_pooling_size'][i][0]

    # Reshape to apply RNN
    features = tf.compat.v1.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
    feature_width = input_shape[2] / width_reduction
    features = tf.compat.v1.reshape(features, tf.compat.v1.stack([tf.compat.v1.cast(feature_width,'int32'), input_shape[0], tf.compat.v1.cast(feature_dim,'int32')]))  # -> [width, batch, features]

    tf.compat.v1.constant(params['img_height'],name='input_height')
    tf.compat.v1.constant(width_reduction,name='width_reduction')

    # Recurrent block
    rnn_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    rnn_outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
        tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_hidden_units), input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [tf.compat.v1.nn.rnn_cell.DropoutWrapper(tf.compat.v1.nn.rnn_cell.BasicLSTMCell(rnn_hidden_units), input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        features,
        dtype=tf.float32,
        time_major=True,
    )
    rnn_outputs = tf.compat.v1.concat(rnn_outputs, 2)

    # Output layer
    logits = tf.compat.v1.layers.dense(
        rnn_outputs,
        params['vocabulary_size'] + 1,  # BLANK
        activation=None,
    )
    tf.compat.v1.add_to_collection("logits",logits) # for restoring purposes

    # CTC Loss computation
    seq_len = tf.compat.v1.placeholder(tf.int32, [None], name='seq_lengths')
    targets = tf.compat.v1.sparse_placeholder(dtype=tf.int32, name='target')
    ctc_loss = tf.compat.v1.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=True)
    loss = tf.compat.v1.reduce_mean(ctc_loss)

    # CTC decoding
    decoded, log_prob = tf.compat.v1.nn.ctc_greedy_decoder(logits, seq_len)

    return input, seq_len, targets, decoded, loss, rnn_keep_prob
