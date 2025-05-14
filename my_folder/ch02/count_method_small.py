import sys
sys.path.append('..')
from common_my.util import *
import numpy as np
import matplotlib.pyplot as plt

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print('co-matrix')
print(C)
W = ppmi(C)
print('PPMI')
print(W)

U, S, V = np.linalg.svd(W)
print('SVD')
print(U)

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.8)
plt.show()