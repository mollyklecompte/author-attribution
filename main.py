import numpy as np
import gensim
import tensorflow as tf

from embeddings import DIM_EMBED, INPUT_LEN
from data_utils import MAX_TWEETS


TARGETS = ['ARR0TT', 'bokwon', 'CAKESDAKILLA', 'chrissyteigen', 'coketweet', 'DoththeDoth', 'efeezyubeezy', 'existentialcoms', 'HE_VALENCIA', 'hrtraulsen', 'janmpdx', 'officialjaden', 'sarahjeong' 'shanley', 'SICKOFWOLVES', 'sideofhail', 'tinynietzsche', 'Tronfucious', 'tylerthecreator']
NUM_TARGETS = len(TARGETS)
CSV_COLUMNS = ['Author', 'Tweet']
LABEL_COLUMN = 'Author'
BATCH_SIZE = None
DEFAULTS = [['null'], ['null']]

filename = 'training.csv'



def read_dataset(file):
    filename = f"tweet_data/data_sets/{file}.csv"
    if file == 'train':
        mode = tf.contrib.learn.ModeKeys.TRAIN
        BATCH_SIZE = MAX_TWEETS / 2 # replace
    else:
        mode = tf.contrib.learn.ModeKeys.EVAL
        BATCH_SIZE = MAX_TWEETS / 4 # replace


    def input_fn():
        input_file_names = tf.train.match_filenames_once(filename)
        filename_queue = tf.train.string_input_producer(input_file_names, shuffle=True)

        # read csv
        reader = tf.TextLineReader()
        _, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE)
        if mode == tf.estimator.ModeKeys.TRAIN:
           value = tf.train.shuffle_batch([value], BATCH_SIZE, capacity=10*batch_size, min_after_dequeue=BATCH_SIZE, enqueue_many=True, allow_smaller_final_batch=False)
        value_column = tf.expand_dims(value, -1)
        columns = tf.decode_csv(value_column, record_defaults=DEFAULTS, field_delim='\\')
        features = dict(zip(CSV_COLUMNS, columns))
        label = features.pop(LABEL_COLUMN)

        # make labels numeric
        table = tf.contrib.lookup.index_table_from_tensor(
                       mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
        labels = table.lookup(label)

        return features, labels
    return input_fn

# load word2vec model and initialize np array with embedding weights for use in tf embedding layer
embeddings_model = gensim.models.Word2Vec.load('models/embeddings')
VOCAB_SIZE = len(embeddings_model.wv.vocab)
embedding_matrix = np.zeros((len(embeddings_model.wv.vocab), DIM_EMBED), dtype=np.float32)
for i in range(len(embeddings_model.wv.vocab)):
    embedding_vector = embeddings_model.wv[embeddings_model.wv.index2word[i]]
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
            padding="VALID",
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
print(h_pool.shape)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
print(h_pool_flat.shape)

with tf.name_scope('fully-connected'):
    logits = tf.layers.dense(h_pool_flat, NUM_TARGETS, activation=None)
    predictions_dict = {
        'source': tf.gather(TARGETS, tf.argmax(logits, 1)),
        'class': tf.argmax(logits, 1),
        'prob': tf.nn.softmax(logits)
    }

print(predictions_dict)



