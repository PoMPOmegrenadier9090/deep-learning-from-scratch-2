import sys
sys.path.append('..')
from common_my.util import *

text = """In a story that's turning heads, a retired U.S. professor has reportedly "married" an AI chatbot named Lucas. Describing him as “a great guy,” she shared her experience of forming a deep emotional connection with the chatbot, highlighting the evolving nature of human-AI relationships."""
corpus, word_to_id, id_to_word = preprocess(text)
print(word_to_id)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('she', word_to_id, id_to_word, C)