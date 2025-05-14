import sys
sys.path.append('..')
sys.path.append('../../')
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')

print(len(corpus))
print(corpus[:30])
print(id_to_word[1])
print(word_to_id['car'])