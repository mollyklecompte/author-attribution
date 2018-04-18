import numpy as np
import gensim
import tensorflow as tf

from embeddings import DIM_EMBED, INPUT_LEN


TARGETS = ['ARR0TT', 'bokwon', 'CAKESDAKILLA', 'chrissyteigen', 'coketweet', 'DoththeDoth', 'efeezyubeezy', 'existentialcoms', 'HE_VALENCIA', 'hrtraulsen', 'janmpdx', 'officialjaden', 'sarahjeong' 'shanley', 'SICKOFWOLVES', 'sideofhail', 'tinynietzsche', 'Tronfucious', 'tylerthecreator']
NUM_TARGETS = len(TARGETS)


# load word2vec model and initialize np array with embedding weights for use in tf embedding layer
model = gensim.models.Word2Vec.load('models/embeddings')
VOCAB_SIZE = len(model.wv.vocab)
embedding_matrix = np.zeros((len(model.wv.vocab), DIM_EMBED), dtype=np.float32)
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

input_x = tf.placeholder(tf.int32, [None, INPUT_LEN], name="input_x")
input_y = tf.placeholder(tf.float32, [None, NUM_TARGETS], name="input_y")

# store embedding matrix as tf variable, to serve as embedding layer

with tf.name_scope("embedding"):
    saved_embeddings = tf.constant(embedding_matrix)
    embedding_layer = tf.Variable(initial_value=saved_embeddings, trainable=False)
    embedded_ngrams = tf.nn.embedding_lookup(embedding_layer, input_x)
    embedded_ngrams_expanded = tf.expand_dims(embedded_ngrams, -1)

# dropout layer
dropout_prob = 0.75
with tf.name_scope("dropout"):
    embedded_ngrams_dropped_out = tf.nn.dropout(embedded_ngrams_expanded, dropout_prob)

# cnn
filter_sizes = [3,4,5]
num_filters = 32
sequence_length = INPUT_LEN

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool"):

        #convolution layer
        filter_shape = [filter_size, DIM_EMBED, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        strides = [1,1,1,1]
        # embedded_ngrams_expanded = tf.expand_dims(embeddings_dropped_out, -1)

        conv = tf.nn.conv2d(
            embedded_ngrams_dropped_out,
            W,
            strides=strides,
            padding="SAME",
            name="conv"
        )
        # conv = tf.contrib.layers.conv2d(embedding_layer, 1, filter_size, padding='VALID')

        # activation function (sigmoid) + bias
        h = tf.nn.sigmoid(tf.nn.bias_add(conv, b), name="sigmoid")

        # max pool over outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides = strides,
            padding="VALID",
            name="pool",
        )
        pooled_outputs.append(pooled)

    # concat pooled outputs
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    print(type(h_pool))

# with tf.name_scope('fully-connected'):


