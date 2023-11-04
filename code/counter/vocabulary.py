class SixtyTwoVocabulary:
    def __init__(self):
        self.beat_token = 48
        self.time_token = {i:i for i in range(48)}
        self.tap_token = {1:49, 2:50, 3:51, 4:52}
        self.hold_token = {1:53, 2:54, 3:55, 4:56}
        self.release_token = {1:57, 2:58, 3:59, 4:60}
        self.bar_token=61

def make_possible_combinations():
    ret = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    if a+b+c+d==0:
                        continue
                    combination = f"{a}{b}{c}{d}"
                    ret.append(combination)        
    return ret

class OneThirtyVocabulary:
    def __init__(self):
        self.beat_token = 48
        self.time_token = {i:i for i in range(48)}
        self.chart_token = {c:49+n for n,c in enumerate(make_possible_combinations())} # 81 total
        self.bar_token = 129


class OneSeventyEightVocabulary:
    def __init__(self):
        self.beat_token = 96
        self.time_token = {i:i for i in range(96)}
        self.chart_token = {c:97+n for n,c in enumerate(make_possible_combinations())} # 81 total
        self.bar_token = 177