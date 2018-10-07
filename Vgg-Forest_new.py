
from skimage import io, transform
from scipy.io import loadmat
import glob
import os
import numpy as np
import tensorflow as tf


# decision tree parameter
DEPTH   = 5                 # Depth of a tree
N_LEAF  = 2 ** (DEPTH + 1)  # Number of leaf node
N_LABEL = 5                # Number of classes
N_TREE  = 8                 # Number of trees (ensemble)
N_BATCH = 16               # Number of data points per mini-batch

# decision picture parameter
w = 224
h = 224
c = 3

# path = 'G:/flower_photos/'
# path = 'C:/Users/Harry/Desktop/flower_photos/'
path = 'E:/Works/HeadgestureData/LFW/'


##################################################
# Load data
##################################################
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        label = [0, 0, 0, 0, 0]
        for im in glob.glob(folder + '/*.jpg'):
            print('reading the images:%s' % (im))
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            label[idx] = 1
            labels.append(label)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.float32)


data, label = read_img(path)

# mess up the order
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

# training set and verification set
ratio = 0.8
s = np.int(num_example * ratio)
trX = data[:s]
trY = label[:s]
teX = data[s:]
teY = label[s:]

trX = trX.reshape(-1, 224, 224, 3)
teX = teX.reshape(-1, 224, 224, 3)


###################################
# Input X, output Y
###################################
X = tf.placeholder("float", [N_BATCH, 224, 224, 3])
Y = tf.placeholder("float", [N_BATCH, N_LABEL])


#######################################
# init weights function
#######################################
def extract_weights(path):
    kernels = []
    data = loadmat(path)
    layers = data['layers']
    for layer in layers[0]:
        layer_type = layer[0]['type'][0][0]
        if layer_type == 'conv':
            kernel, bias = layer[0]['weights'][0][0]
            kernels.append(kernel)
    return kernels


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_prob_weights(shape, minval=-5, maxval=5):
    return tf.Variable(tf.random_uniform(shape, minval, maxval))


###########################
# Convolution function
###########################
global kernels
kernels = extract_weights('vgg-face.mat')


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p, wp):
    with tf.name_scope(name) as scope:
        kernel = kernels.copy()
        kernel = tf.Variable(kernel[wp], trainable=False, name="conv_kernel")
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=False, name="conv_b")
        # biases =bias[wp]
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation



###########################
# Pooling function
###########################
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


###################################
# Full connection function
###################################
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 trainable=True)
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), trainable=True, name="b")
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


#############################################
# Make convolution and pooling function
#############################################
def inference_op(input_op, keep_prob):
    p = []
    # assume input_op shape is 224x224x3

    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p, wp=0)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p, wp=1)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)
    # pool1 = tf.nn.dropout(pool1, keep_prob)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p, wp=2)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p, wp=3)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)
    # pool2 = tf.nn.dropout(pool2, keep_prob)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p, wp=4)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p, wp=5)
    conv3_3 = conv_op(conv3_2,  name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p, wp=6)
    # conv3_4 = conv_op(conv3_3,  name="conv3_4", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3,   name="pool3",   kh=2, kw=2, dh=2, dw=2)
    # pool3 = tf.nn.dropout(pool3, keep_prob)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=7)
    conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=8)
    conv4_3 = conv_op(conv4_2,  name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=9)
    # conv4_4 = conv_op(conv4_3,  name="conv4_4", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)
    # pool4 = tf.nn.dropout(pool4, keep_prob)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=10)
    conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=11)
    conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p, wp=12)
    # conv5_4 = conv_op(conv5_3,  name="conv5_4", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)

    return pool5    # 7*7*512

    # # flatten
    # shp = pool5.get_shape()
    # flattened_shape = shp[1].value * shp[2].value * shp[3].value
    # resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")
    #
    # # fully connected
    # fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    # fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")
    #
    # fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    # fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")
    #
    # fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    # softmax = tf.nn.softmax(fc8)
    # predictions = tf.argmax(softmax, 1)
    # return predictions, softmax, fc8, p


#############################
# Define model function
#############################
def model(X, w_e, w_d_e, w_l_e, p_keep_conv, p_keep_hidden):

    # assertion --judge a formula true or false
    assert (len(w_e) == len(w_d_e))
    assert (len(w_e) == len(w_l_e))

    pool = inference_op(X, p_keep_conv)      # net output

    pool = tf.reshape(pool, [-1, w_e[0].get_shape().as_list()[0]])
    pool = tf.nn.dropout(pool, p_keep_conv)

    decision_p_e = []
    leaf_p_e = []
    for w, w_d, w_l in zip(w_e, w_d_e, w_l_e):
        op = tf.nn.relu(tf.matmul(pool, w))
        op = tf.nn.dropout(op, p_keep_hidden)


        decision_p = tf.nn.sigmoid(tf.matmul(op, w_d))
        leaf_p = tf.nn.softmax(w_l)

        decision_p_e.append(decision_p)
        leaf_p_e.append(leaf_p)

    return decision_p_e, leaf_p_e


##################################################
# Load data
##################################################
# mnist = input_data.read_data_sets("MNIST/", one_hot=True)
# trX, trY = mnist.train.images, mnist.train.labels
# teX, teY = mnist.test.images, mnist.test.labels
# trX = trX.reshape(-1, 224, 224, 3)
# teX = teX.reshape(-1, 224, 224, 3)


###################################
# Input X, output Y
###################################
# X = tf.placeholder("float", [N_BATCH, 224, 224, 3])
# Y = tf.placeholder("float", [N_BATCH, N_LABEL])


############################################
# Initialize decision tree weights
############################################
w_ensemble = []
w_d_ensemble = []
w_l_ensemble = []

for i in range(N_TREE):
       w_ensemble.append(init_weights([512 * 7 * 7, 4096]))
       w_d_ensemble.append(init_prob_weights([4096, N_LEAF], -1, 1))
       w_l_ensemble.append(init_prob_weights([N_LEAF, N_LABEL], -2, 2))

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")


###########################################
# Define a fully differentiable deep-ndf
###########################################
# With the probability decision_p, route a sample to the right branch


decision_p_e, leaf_p_e = model(X, w_ensemble, w_d_ensemble,
                               w_l_ensemble, p_keep_conv, p_keep_hidden)


flat_decision_p_e = []

# iterate over each tree
for decision_p in decision_p_e:
    # Compute the complement of d, which is 1 - d
    # where d is the sigmoid of fully connected output
    decision_p_comp = tf.subtract(tf.ones_like(decision_p), decision_p)

    # Concatenate both d, 1-d
    decision_p_pack = tf.stack([decision_p, decision_p_comp])

    # Flatten or vectorize the decision probabilities for efficient indexing
    flat_decision_p = tf.reshape(decision_p_pack, [-1])
    flat_decision_p_e.append(flat_decision_p)

# 0 index of each data instance in a mini-batch
batch_0_indices = \
    tf.tile(tf.expand_dims(tf.range(0, N_BATCH * N_LEAF, N_LEAF), 1),
            [1, N_LEAF])


###############################################################################
# The routing probability computation
#
# We will create a routing probability matrix \mu. First, we will initialize
# \mu using the root node d, 1-d. To efficiently implement this routing, we
# will create a giant vector (matrix) that contains all d and 1-d from all
# decision nodes. The matrix version of that is decision_p_pack and vectorized
# version is flat_decision_p.
#
# The suffix `_e` indicates an ensemble. i.e. concatenation of all responsens
# from trees.
#
# For depth = 2 tree, the routing probability for each leaf node can be easily
# compute by multiplying the following vectors elementwise.
# \mu =       [d_0,   d_0,   d_0,   d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
# \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
# \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]
#
# Tree indexing
#      0
#    1   2
#   3 4 5 6
##############################################################################

in_repeat = int(N_LEAF / 2)
out_repeat = int(N_BATCH)

# Let N_BATCH * N_LEAF be N_D. flat_decision_p[N_D] will return 1-d of the
# first root node in the first tree.
batch_complement_indices = \
    np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
             * out_repeat).reshape(N_BATCH, N_LEAF)

# First define the routing probabilities d for root nodes
mu_e = []

# iterate over each tree
for i, flat_decision_p in enumerate(flat_decision_p_e):
    mu = tf.gather(flat_decision_p,
                   tf.add(batch_0_indices, batch_complement_indices))
    mu_e.append(mu)

# from the second layer to the last layer, we make the decision nodes
for d in range(1, DEPTH + 1):
    indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
    tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                      [1, 2 ** (DEPTH - d + 1)]), [1, -1])
    batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, [N_BATCH, 1]))

    in_repeat = int(in_repeat / 2)
    out_repeat = int(out_repeat * 2)

    # Again define the indices that picks d and 1-d for the node
    batch_complement_indices = \
        np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
                 * out_repeat).reshape(N_BATCH, N_LEAF)

    mu_e_update = []
    for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
        mu = tf.multiply(mu, tf.gather(flat_decision_p,
                                  tf.add(batch_indices, batch_complement_indices)))
        mu_e_update.append(mu)

    mu_e = mu_e_update


##################################################
# Define p(y|x)
##################################################
py_x_e = []
for mu, leaf_p in zip(mu_e, leaf_p_e):
    # average all the leaf p
    py_x_tree = tf.reduce_mean(
        tf.multiply(tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABEL]),
               tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1])), 1)
    py_x_e.append(py_x_tree)

py_x_e = tf.stack(py_x_e)
py_x = tf.reduce_mean(py_x_e, 0)


##################################################
# Define cost and optimization method
##################################################

# cross entropy loss
cost = tf.reduce_mean(-tf.multiply(tf.log(py_x), Y))

# cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(py_x, Y))
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict = tf.argmax(py_x, 1)


###################################################
# Train and Test
###################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for i in range(100):
    # One epoch
    for start, end in zip(range(0, len(trX), N_BATCH), range(N_BATCH, len(trX), N_BATCH)):
        sess.run(train_step, feed_dict={X: trX[start:end], Y: trY[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})

    # Result on the test set
    results = []
    for start, end in zip(range(0, len(teX), N_BATCH), range(N_BATCH, len(teX), N_BATCH)):
        results.extend(np.argmax(teY[start:end], axis=1) ==
            sess.run(predict, feed_dict={X: teX[start:end], p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0}))

    print('Epoch: %d, Test Accuracy: %f' % (i + 1, np.mean(results)))


# Save log
writer = tf.summary.FileWriter(path, tf.get_default_graph())
writer.close()

# Save model
saver.save(sess, path + 'model/model.ckpt')
