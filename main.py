import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10)
        )
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

def main():
    train, test = datasets.get_mnist()
    train_iter = iterators.SerialIterator(train, batch_size=100)
    test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

    model = L.Classifier(MLP())
    model.to_gpu()
    optimizer = optimizers.SGD()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == "__main__":
    main()
