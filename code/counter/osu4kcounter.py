import os
import numpy as np
from .vocabulary import OneThirtyVocabulary, OneSeventyEightVocabulary


class OneSeventyEightCounter():
    def __init__(self):
        self.voca = OneSeventyEightVocabulary()
        self.reset()

    def reset(self):
        self.all_notes = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0

    def stats_with(self, other_number):
        if self.true_positive + other_number == 0:
            return 0
        return self.true_positive / (self.true_positive + other_number)

    def recall(self):
        return self.stats_with(self.false_negative)

    def precision(self):
        return self.stats_with(self.false_positive)

    def f1(self):
        try:
            return 2 * self.precision() * self.recall() / (self.precision() + self.recall())
        except ZeroDivisionError:
            return 0

    def decipher_token(self, token):
        token -= 96
        ret = [(token//27),(token//9)%3,(token//3)%3,token%3]
        ret = [str(x) for x in ret]
        return "".join(ret)
    
class Osu4kTwoBeatOnsetCounter(OneSeventyEightCounter):
    def __init__(self):
        super().__init__()

    def update(self, ref, pred):
        board_ref = [ "0000" for _ in range(96) ]
        board_pred = [ "0000" for _ in range(96)]
        mode = 0
        for board, tokens in [(board_ref, ref),(board_pred, pred)]:
            if len(tokens) == 0:
                continue
            for token in tokens:
                token = int(token)
                assert token != self.voca.beat_token
                if token == self.voca.bar_token:
                    break
                elif token in self.voca.time_token.keys():
                    mode = int(token)
                elif token in self.voca.chart_token.values():
                    board[mode] = self.decipher_token(int(token))

        for keys_ref, keys_pred in zip(board_ref, board_pred):
            assert len(keys_ref) == len(keys_pred)
            note_ref, note_pred = int(keys_ref), int(keys_pred)
            if note_ref == 0 and note_pred == 0:
                continue
            elif note_ref == 0 and note_pred != 0:
                self.false_positive += 1
            elif note_ref != 0 and note_pred == 0:
                self.false_negative += 1
            elif note_ref != 0 and note_pred != 0:
                self.true_positive += 1

class Osu4kTwoBeatTimingCounter(OneSeventyEightCounter):
    def __init__(self):
        super().__init__()

    def update(self, ref, pred):
        board_ref = [ 0 for _ in range(96) ]
        board_pred = [ 0 for _ in range(96)]
        mode = 0
        for board, tokens in [(board_ref, ref),(board_pred, pred)]:
            if len(tokens) == 0:
                continue
            for token in tokens:
                token = int(token)
                assert token != self.voca.beat_token
                if token == self.voca.bar_token:
                    break
                elif token in self.voca.time_token.keys():
                    board[int(token)] = 1

        for note_ref, note_pred in zip(board_ref, board_pred):
            if note_ref == 0 and note_pred == 0:
                continue
            elif note_ref == 0 and note_pred != 0:
                self.false_positive += 1
            elif note_ref != 0 and note_pred == 0:
                self.false_negative += 1
            elif note_ref != 0 and note_pred != 0:
                self.true_positive += 1

class Osu4kTwoBeatNotesCounter(OneSeventyEightCounter):
    def __init__(self):
        super().__init__()

    def update(self, ref, pred):
        board_ref = [ "0000" for _ in range(96) ]
        board_pred = [ "0000" for _ in range(96)]
        mode = 0
        for board, tokens in [(board_ref, ref),(board_pred, pred)]:
            if len(tokens) == 0:
                continue
            for token in tokens:
                token = int(token)
                assert token != self.voca.beat_token
                if token == self.voca.bar_token:
                    break
                elif token in self.voca.time_token.keys():
                    mode = int(token)
                elif token in self.voca.chart_token.values():
                    board[mode] = self.decipher_token(int(token))
        
        for keys_ref, keys_pred in zip(board_ref, board_pred):
            assert len(keys_ref) == len(keys_pred)
            for note_ref, note_pred in zip(keys_ref, keys_pred):
                note_ref, note_pred = int(note_ref), int(note_pred)
                if note_ref == 0 and note_pred == 0:
                    continue
                elif note_ref == 0 and note_pred != 0:
                    self.false_positive += 1
                elif note_ref != 0 and note_pred == 0:
                    self.false_negative += 1
                elif note_ref != 0 and note_pred != 0:
                    self.true_positive += 1




def test():
    counter = Osu4kTwoBeatCounter()
    counter.update([0, 97, 16, 98, 48, 123, 72, 167], [0, 97, 16, 98, 48, 124, 72, 173])
    print(counter.precision())
    print(counter.recall())
    print(counter.f1())

if __name__=="__main__":
    test()
