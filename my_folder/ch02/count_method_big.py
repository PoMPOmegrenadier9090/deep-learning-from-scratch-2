import sys
sys.path.append('..')
sys.path.append('../../')
import numpy as np
from dataset import ptb
from common_my.util import *

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

print('counting co-occurence')
C = create_co_matrix(corpus,vocab_size,window_size)

print('calculating PPMI')
W = ppmi(C, verbose=True)

print('calculating SVD ....')
try:
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    U, S, V = np.linalg.svd(W)

word_vecs = U[:,:wordvec_size]

queries = ['you', 'year', 'car', 'toyota']
for query in queries:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)