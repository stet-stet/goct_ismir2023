import os
import numpy as np
import torch
from .osu4kcounter import OneSeventyEightCounter
from .vocabulary import OneThirtyVocabulary, OneSeventyEightVocabulary


class Osu4kCLSTMCounter(OneSeventyEightCounter):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def update(self, pred, ref):
        """
        ref, pred: must be of identical shape.

        ref: an array of 1s and 0s.
        pred: an array of possibility values between 0 and 1. 
        """
        pred = (pred > self.threshold)
        ref = (ref == 1)

        self.true_positive += float(torch.sum(ref & pred))
        self.false_positive += float(torch.sum((~ref) & pred))
        self.false_negative += float(torch.sum(ref & (~pred)))


def test():
    counter = Osu4kCLSTMCounter(0.5)
    counter.update(torch.tensor([[0, 0, 0, 1], [1, 1, 1, 1]]), torch.tensor([[0, 0, 0, 1], [1, 0, 1, 1]]))
    print(counter.precision())
    print(counter.recall())
    print(counter.f1())

if __name__=="__main__":
    test()