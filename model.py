import tensorflow as tf
import input_cs
import numpy as np

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.


def create_conv_layer(x, n_in, n_out):
    W = tf.get_variable("W", shape=[5, 5, n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b = tf.get_variable("b", shape=[n_out], dtype=tf.float32, initializer=tf.zeros_initializer())

    z = tf.nn.conv2d(x, filter=W, strides=[1,1,1,1], padding="SAME")
    z = tf.nn.bias_add(z, b)

    z = tf.nn.relu(z)

    return tf.nn.max_pool(z, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")

def create_fully_connected_layer(x, n_in, n_out, relu):
    W = tf.get_variable("W", shape=[n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=[n_out], dtype=tf.float32, initializer=tf.zeros_initializer())

    xW = tf.matmul(x, W)
    z = tf.nn.bias_add(xW, b)

    if relu:
        return tf.nn.relu(z)
    else:
        return z

def build_model(images, num_classes):
    """Build the model

    :param images: Input image placeholder
    :param num_classes: Nbr of final output classes
    :return: Output of final fc-layer
    """
    #####Insert your code here for subtask 1e#####
    # It might be useful to define helper functions which add a layer of type needed
    # If you define such as function, remember that multiple variables with the same name will result in an error
    # To this end you may want to use with tf.variable_scope(name) to define a named scope for each layer
    # This way, you get a less cluttered visualization of the graph in tensorboard and debugging may be easier in tfdbg
    x = images
    with tf.variable_scope("Conv_1"):
        x = create_conv_layer(x, n_in=3, n_out=24)
    with tf.variable_scope("Conv_2"):
        x = create_conv_layer(x, n_in=24, n_out=32)
    with tf.variable_scope("Conv_3"):
        x = create_conv_layer(x, n_in=32, n_out=50)

    #####Insert your code here for subtask 1f#####
    # Add fc-classifictaion-layers

    height = x.get_shape()[1]
    width = x.get_shape()[2]
    n_features = x.get_shape()[3]
    n_features_collapsed = height * width * n_features
    x = tf.reshape(x, [-1,n_features_collapsed])

    with tf.variable_scope("Full_1"):
        x = create_fully_connected_layer(x, n_in=n_features_collapsed, n_out=100, relu=True)
    with tf.variable_scope("Full_2"):
        x = create_fully_connected_layer(x, n_in=100, n_out=50, relu=True)
    with tf.variable_scope("Full_3"):
        x = create_fully_connected_layer(x, n_in=50, n_out=num_classes, relu=False)

    softmax_logits = x
    
    return softmax_logits


def loss(logits, labels):
    """ Add cross entropy loss to the graph

    :param logits: Linear logits for spare_softmax
    :param labels: Ground truth labels
    :return: Mean cross entropy loss, cross entropy loss for every training example seperately
            (used for validation purposes)
    """

    #####Insert your code here for subtask 1g#####
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    _add_loss_summaries(cross_entropy_mean)

    return cross_entropy_mean, cross_entropy


def get_train_op_for_loss(loss, global_tr_step, batch_size, initial_lr ):
    """Add training operation (gradient descent for given loss) to the graph

    :param loss: Loss value
    :param global_tr_step: Tensorflow global training step
    :param batch_size: Batch size
    :param initial_lr: Initial learning rate
    :return: Training operation
    """

    #####Insert your code here for subtask 1h#####
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=initial_lr)
    train_op = optimizer.minimize(loss=loss, global_step=global_tr_step)

    return train_op


def _add_loss_summaries(loss):
    tf.summary.scalar('Loss', loss)


if __name__ == "__main__":
    images = tf.placeholder(dtype=tf.float32, shape=[None] + input_cs.IMAGE_SIZE + [3])
    logits = build_model(images, input_cs.NUM_CLASSES)
