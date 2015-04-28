__author__ = 'DY'

from main import *
from cs224d.datasets.data_utils import *

# Implement your skip-gram and CBOW models here

# Interface to the dataset for negative sampling
# dataset = type('dummy', (), {})()
class DummyDataset:
    def __init__(self):
        self._tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    def tokens(self):
        return self._tokens

    def sampleTokenIdx(self):
        if hasattr(self, "_sampleTokenIdx") and self._sampleTokenIdx:
            return self._sampleTokenIdx
        self._sampleTokenIdx = random.randint(0,4)
        return self._sampleTokenIdx
    def getRandomContext(self, C):
        if hasattr(self, "_getRandomContext") and self._getRandomContext:
            return self._getRandomContext
        tokens = ["a", "b", "c", "d", "e"]
        self._getRandomContext = [tokens[random.randint(0, 4)] for i in xrange(2*C+1)]
        return self._getRandomContext
dataset = DummyDataset()
random.seed(31415)
np.random.seed(9265)

# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    """ Row normalization function """
    ### YOUR CODE HERE
    normLen = np.sqrt(np.sum(x ** 2, axis=1));
    x = x / lift(normLen).T
    ### END YOUR CODE
    return x
# Test this function
print "=== For autograder ==="
print normalizeRows(np.array([[3.0,4.0],[1, 2]]))  # the result should be [[0.6, 0.8], [0.4472, 0.8944]]

wordVectors = normalizeRows(np.random.randn(10,3))

def softmaxCostAndGradient(predicted, target, outputVectors):
    """ Softmax cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, assuming the softmax prediction function and cross      #
    # entropy loss.                                                   #
    # Inputs:                                                         #
    #   - predicted: numpy ndarray, predicted word vector             #
    #   - target: integer, the index of the target word               #
    #   - outputVectors: "output" vectors for all tokens              #
    # Outputs:                                                        #
    #   - cost: cross entropy cost for the softmax word prediction    #
    #   - gradPred: the gradient with respect to the predicted word   #
    #           vector                                                #
    #   - grad: the gradient with respect to all the other word       #
    #           vectors                                               #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE

    costN = predicted.dot(outputVectors.T)
    costN = np.exp(costN)
    costNNorm = costN / np.sum(costN)
    cost = np.log(costNNorm[target])

    grad = lift(-costNNorm).T.dot(lift(predicted))
    grad[target] += predicted

    gradPred = outputVectors[target] - costN.dot(outputVectors) / np.sum(costN)

    ### END YOUR CODE

    return cost, gradPred, grad

print "==== Gradient check for softmaxCostAndGradient ==="

def softmaxCostAndGradient_wrapper(wordVectors):
    N = wordVectors.shape[0] / 2
    cost, gradPred, gradOut = softmaxCostAndGradient(wordVectors[0], 0, wordVectors[N:,:])
    grad = np.zeros(wordVectors.shape)
    grad[0] = gradPred
    grad[N:,:] = gradOut
    return cost, grad
gradcheck_naive(softmaxCostAndGradient_wrapper, wordVectors)


def negSamplingCostAndGradient(predicted, center, outputVectors, K=10):
    """ Negative sampling cost function for word2vec models """
    ###################################################################
    # Implement the cost and gradients for one predicted word vector  #
    # and one target word vector as a building block for word2vec     #
    # models, using the negative sampling technique. K is the sample  #
    # size. You might want to use dataset.sampleTokenIdx() to sample  #
    # a random word index.                                            #
    # Input/Output Specifications: same as softmaxCostAndGradient     #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE

    cost = predicted.dot(outputVectors[center])
    cost = 1 / (1 + np.exp(-cost))
    gradPred = (1 - cost) * outputVectors[center]

    grad = np.zeros(outputVectors.shape)

    grad[center] = (1 - cost) * predicted

    cost = np.log(cost)

    for i in range(K):
        dummy_sample_token_idx = dataset.sampleTokenIdx()
        negVectors = outputVectors[dummy_sample_token_idx]
        negCost = -negVectors.dot(predicted)
        negCost = 1 / (1 + np.exp(-negCost))
        gradPred += (1 - negCost) * -outputVectors[dummy_sample_token_idx]
        negGrad = (1 - negCost) * -predicted
        grad[dummy_sample_token_idx] += negGrad
        cost += np.log(negCost)

    ### END YOUR CODE

    return cost, gradPred, grad

print "==== Gradient check for negSamplingCostAndGradient ==="

def negSamplingCostAndGradient_wrapper(wordVectors):
    N = wordVectors.shape[0] / 2
    cost, gradPred, gradOut = negSamplingCostAndGradient(wordVectors[0], 0, wordVectors[N:,:])
    grad = np.zeros(wordVectors.shape)
    grad[0] = gradPred
    grad[N:,:] = gradOut
    return cost, grad
gradcheck_naive(negSamplingCostAndGradient_wrapper, wordVectors)

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """
    ###################################################################
    # Implement the skip-gram model in this function.                 #
    # Inputs:                                                         #
    #   - currrentWord: a string of the current center word           #
    #   - C: integer, context size                                    #
    #   - contextWords: list of 2*C strings, the context words        #
    #   - tokens: a dictionary that maps words to their indices in    #
    #             the word vector list                                #
    #   - inputVectors: "input" word vectors for all tokens           #
    #   - outputVectors: "output" word vectors for all tokens         #
    #   - word2vecCostAndGradient: the cost and gradient function for #
    #             a prediction vector given the target word vectors,  #
    #             could be one of the two cost functions you          #
    #             implemented above                                   #
    # Outputs:                                                        #
    #   - cost: the cost function value for the skip-gram model       #
    #   - grad: the gradient with respect to the word vectors         #
    # We will not provide starter code for this function, but feel    #
    # free to reference the code you previously wrote for this        #
    # assignment!                                                     #
    ###################################################################

    ### YOUR CODE HERE
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    for i in range(2*C):
        target = tokens[contextWords[i]]
        predicted = inputVectors[tokens[currentWord]]
        costDelta, gradPred, gradDelta = word2vecCostAndGradient(predicted, target, outputVectors)
        cost += costDelta
        gradIn[tokens[currentWord]] += gradPred
        gradOut += gradDelta

    ### END YOUR CODE

    return cost / 2 / C, gradIn / 2 / C, gradOut / 2 / C

print "==== Gradient check for skip-gram ===="
def word2vec_sgd_wrapper(word2vecModel, C, word2vecCostAndGradient, wordVectors):
    batchsize = 1
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0] / 2
    inputVectors = wordVectors[:N,:]
    outputVectors = wordVectors[N:,:]
    for i in xrange(batchsize):
        context = dataset.getRandomContext(C)
        c, gin, gout = word2vecModel(context[C], C, context[:C] + context[C+1:], dataset.tokens(), inputVectors, outputVectors, word2vecCostAndGradient)
        cost += c / batchsize
        grad[:N, :] += gin / batchsize
        grad[N:, :] += gout / batchsize

    return cost, grad
gradcheck_naive(lambda wordVectors: word2vec_sgd_wrapper(skipgram, 5, softmaxCostAndGradient, wordVectors), wordVectors)
gradcheck_naive(lambda wordVectors: word2vec_sgd_wrapper(skipgram, 5, negSamplingCostAndGradient, wordVectors), wordVectors)

# Now, implement SGD

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 10000

import glob
import os.path as op

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        return st, np.load("saved_params_%d.npy" % st)
    else:
        return st, None

def save_params(iter, params):
    np.tensordot()
    np.reshape()
    np.dot()
    np.save("saved_params_%d.npy" % iter, params)

def sgd(f, x0, step, iterations, postprocessing = None, useSaved = False):
    """ Stochastic Gradient Descent """
    ###################################################################
    # Implement the stochastic gradient descent method in this        #
    # function.                                                       #
    # Inputs:                                                         #
    #   - f: the function to optimize, it should take a single        #
    #        argument and yield two outputs, a cost and the gradient  #
    #        with respect to the arguments                            #
    #   - x0: the initial point to start SGD from                     #
    #   - step: the step size for SGD                                 #
    #   - iterations: total iterations to run SGD for                 #
    #   - postprocessing: postprocessing function for the parameters  #
    #        if necessary. In the case of word2vec we will need to    #
    #        normalize the word vectors to have unit length.          #
    # Output:                                                         #
    #   - x: the parameter value after SGD finishes                   #
    ###################################################################

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 50000

    if useSaved:
        start_iter, oldx = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in xrange(start_iter + 1, iterations + 1):
        ### YOUR CODE HERE
        ### Don't forget to apply the postprocessing after every iteration!
        ### You might want to print the progress every few iterations.
        random_context = dataset.getRandomContext()
        cost, xdelta = f(x)
        x -= step * xdelta
        x = postprocessing(x)
        ### END YOUR CODE

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x

# Load some data and initialize word vectors
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5
# Train word vectors (this could take a while!)

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
wordVectors = normalizeRows(np.random.randn(nWords * 2, dimVectors))
wordVectors0 = sgd(lambda wordVectors: word2vec_sgd_wrapper(skipgram, C, softmaxCostAndGradient, wordVectors), wordVectors, 10.0, 200000, normalizeRows, True)

# just use the output vectors
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:]) / 2.0

print "\n=== For autograder ==="
checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
checkIdx = [tokens[word] for word in checkWords]
checkVecs = wordVectors[checkIdx, :]
print checkVecs

# Visualize the word vectors you trained
_, wordVectors0 = load_saved_params()
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:]) / 2.0
visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "warm", "enjoyable", "boring", "bad", "garbage", "waste", "disaster", "dumb", "embarrassment", "annoying", "disgusting"]
visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
covariance = visualizeVecs.T.dot(visualizeVecs)
U,S,V = np.linalg.svd(covariance)
coord = (visualizeVecs - np.mean(visualizeVecs, axis=0)).dot(U[:,0:2])

for i in xrange(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.show()