import os
import tqdm
import numpy as np
from utils import *
from model import *
#from keras.backend.tensorflow_backend import set_session

train_pair, test_pair, adj_matrix, r_index, r_val, adj_features, rel_features = load_data('data/en_de_15k_V1/mapping/0_3/',train_ratio=0.3)
adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data


node_size = adj_features.shape[1]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
batch_size = node_size
model, get_emb = get_model(lr=0.001, dropout_rate=0.30, node_size=node_size, rel_size=rel_size, n_attn_heads=2,
                          depth=2, gamma=3, node_hidden=100, rel_hidden=100, triple_size=triple_size, batch_size=batch_size)
model.summary()

def get_train_set(batch_size,train_pair):
    negative_ratio = batch_size // len(train_pair) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_pair, axis=0), axis=0, repeats=negative_ratio),newshape=(-1, 2))
    np.random.shuffle(train_set); train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set,np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set

def test():
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    se_vec = get_emb.predict_on_batch(inputs)
    get_hits(se_vec,test_pair)
    print()
    return se_vec

for epoch in tqdm.trange(5000):
    train_set = get_train_set(batch_size, train_pair)
    inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_set]
    inputs = [np.expand_dims(item, axis=0) for item in inputs]
    model.train_on_batch(inputs, np.zeros((1, 1)))
    if (epoch%1000 == 999):
        test()