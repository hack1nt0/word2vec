__author__ = 'DY'

import numpy as np

class XNN():
    def __init__(self, objNeuron):
        self.root = objNeuron
        self.topo = []
        self.topoSort(objNeuron)
        # self.topo.reverse()

    def topoSort(self, tail):
        if tail.pre is not None:
            for preChd in tail.pre:
                self.topoSort(preChd)
        self.topo.append(self)

    def forwardProp(self):
        for neuron in self.topo:
            neuron.getA()
        return self.root.a

    def backwardProp(self, updGrad):
        for neuron in self.topo.reverse():
            neuron.getGrad(updGrad)




class CrossEntropyNeuron():
    def __init__(self, pre, label):
        self.p = label
        self.pre = pre.a()

    # recursively(top-down) calculate the activation
    def getA(self):
        cost = -np.dot(self.p, np.log(self.pre.getA()))
        return cost
    def getGrad(self, updGrad):
        gradpa = -self.p / self.q
        return gradpa

class SoftmaxNeuron():
    def __init__(self, pre, N, M, next):
        self.next = next
        self.pre = pre
        self.W = np.random.rand(N, M)
        self.b = np.random.rand(M)
        self.gradW = np.random.rand(N, M)
        self.gradb = np.random.rand(M)

    def getA(self):
        z = np.dot(self.pre.getA(), self.W)
        a = np.exp(z)
        a = a / np.sum(a)
        self.a = a #todo
        return a

    def getGrad(self, updGrad):
        grada = self.next.grad()
        gradz = np.dot(grada, np.dot(self.a, -self.a) + np.eye(self.a.size, self.a.size))
        self.gradW = np.outer(self.pre.a, gradz)
        self.gradb = gradz
        gradpa = np.dot(gradz, self.W.T)
        return gradpa

class SigmoidNeuron():
    def __init__(self, pre, N, M, next):
        self.next = next
        self.pre = pre
        self.W = np.random.rand(N, M)
        self.b = np.random.rand(M)
        self.gradW = np.random.rand(N, M)
        self.gradb = np.random.rand(M)

    def getA(self):
        z = np.dot(self.pre.getA(), self.W)
        a = 1. / 1. + np.exp(z)
        self.a = a
        return a

    def getGrad(self, updGrad):
        grada = self.next.grad()
        gradz = grada * (1. - grada)
        gradW = np.outer(self.pre.a, gradz)
        gradb = gradz
        updGrad(self.W, gradW)
        updGrad(self.b, gradb)
        self.gradpa = np.dot(gradz, self.W.T)

