import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import chainer

from sobamchan.sobamchan_chainer import Model
from sobamchan.sobamchan_chainer_link import PreTrainedEmbedId
from sobamchan.sobamchan_vocabulary import Vocabulary

class CNN(Model):

    def __init__(self, class_n, d, vocab, fpath):
        embed_learn=PreTrainedEmbedId(len(vocab),d,vocab,fpath,False)
        super(CNN, self).__init__(
            embed_learn,
            conv_f3=L.Convolution2D(2, 100, (3, d)),
            conv_f4=L.Convolution2D(2, 100, (4, d)),
            conv_f5=L.Convolution2D(2, 100, (5, d)),
        )
        self.embedW = embed_learn[0].W
        self.embed_static = F.embed_id

    def __call__(self, x, t, train=True):
        y = self.fwd(x, train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def fwd(self, x, train):
        embedW = self.embedW
        embed_h1 = self.embed_learn(x)
        embed_h2 = self.embed_static(x, embedW)
        b, embed_h, embed_w = embed_h1.shape
