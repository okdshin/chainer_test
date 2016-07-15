import numpy as np
import chainer
#from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import cuda, Function, gradient_check, utils, Variable
from chainer import datasets, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

def main():
    f = L.Linear(3, 2)
    f.zerograds()
    x = Variable(np.array([[1,2,3], [4,5,6]], dtype=np.float32))
    y = f(x)
    print(y.data)
    print(f.W.data)
    y.grad = np.ones((2,2), dtype=np.float32)
    y.backward()
    print(f.W.grad)
    print(f.b.grad)

    optimizer = optimizers.SGD()
    optimizer.setup(f)
    train, test = datasets.get_mnist()
    print(len(train), len(train[0]))
    print(train[0][1])

if __name__ == "__main__":
    main()
