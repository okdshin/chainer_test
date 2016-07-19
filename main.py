import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class DeepCNet(ChainList):
    def __init__(self, l, k, output_dim):
        cnns = []
        cnns.append(L.Convolution2D(in_channels=3, out_channels=k,
                                    ksize=3, pad=1, use_cudnn=True))
        for i in range(l):
            cnns.append(L.Convolution2D(in_channels=(i+1)*k, out_channels=(i+2)*k,
                                        ksize=2, pad=1, use_cudnn=True))
        fc=L.Linear((l+1)*k, output_dim)
        super(DeepCNet, self).__init__(*cnns, fc)

    def __call__(self, x):
        y = x
        for conv in self[:-1]:
            y = F.relu(F.max_pooling_2d(conv(y), ksize=2, stride=2, cover_all=False))
        y = F.relu(self[-1](y))
        return y

def main():
    train, test = datasets.get_cifar10(ndim=3)
    train_iter = iterators.SerialIterator(train, batch_size=100)
    test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

    model = L.Classifier(DeepCNet(5, 300, 10))
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
