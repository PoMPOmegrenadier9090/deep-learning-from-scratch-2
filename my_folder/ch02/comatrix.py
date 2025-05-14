import sys
sys.path.append('..')
import numpy as np
from common.util import *

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)

print(id_to_word)

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

print(C)

c_you = C[word_to_id['you']]
c_i = C[word_to_id['i']]

print(cos_similarity(c_you, c_i))