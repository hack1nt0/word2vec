__author__ = 'dy'
from gensim.models.word2vec import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from cs224d.datasets.data_utils import *

dataset = StanfordSentiment()
sentences = dataset.sentences()

model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
# model.save_word2vec_format("baseline.model")

print "\n=== For autograder ==="
checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
# checkIdx = [model.vocab[word].index for word in checkWords]
# checkVecs = model[checkIdx, :]
checkVecs = np.array([model[w] for w in checkWords])
print checkVecs

# Visualize the word vectors you trained
# model = model.load_word2vec_format("baseline.model")

visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "warm", "enjoyable", "boring", "bad", "garbage", "waste", "disaster", "dumb", "embarrassment", "annoying", "disgusting"]
visualizeVecs = np.array([model[w] for w in visualizeWords])

import visualizing as vs

vs.visualize(visualizeVecs, visualizeWords, "baseline")
