import numpy as np
import gensim
import tensorflow as tf

from .embeddings import dim_embed

# load word2vec model and initialize np array with embedding weights for use in tf embedding layer
model = gensim.models.Word2Vec.load('models/embeddings')
embedding_matrix = np.zeros((len(model.wv.vocab), dim_embed))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# store embedding matrix as tf variable, to serve as embedding layer
saved_embeddings = tf.constant(embedding_matrix)
embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)
