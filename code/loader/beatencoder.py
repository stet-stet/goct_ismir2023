"""
given a consecutive subset of notes (given in a [[bar, quant, pos],totl_beat,time(s),encoded_hits] format),
transforms this into a sequence of chart tokens.

THE LENGTH MUST BE ONE BEAT.

Rather than as a single construct, this will be used as part of another encoder, which will load an arbitrary number of beats.

Written by Jayeon Yi (stet-stet)
"""

import abc
from .vocabulary import SixtyTwoVocabulary, OneThirtyVocabulary, OneSeventyEightVocabulary

def round_to_nearest_48th(num):
    return round(num*48)/48

def get_quantization(num):
    assert num<1 and num>=0
    return round(48*num)

class BeatEncoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def segmentation_policy(self,chart, current_index):
        pass
        """
        seg policy : (chart, current_index) -> next_index
        trans policy : (chart[current_index:next_index]) -> token for chart[current_index:next_index].
        """

    @abc.abstractmethod
    def translation_policy(self,segmented_chart):
        pass 

    @abc.abstractmethod
    def encode(self,consecutive_notes):
        pass

    @abc.abstractmethod
    def pretty_print(self,encoded_tokens):
        """No, you're not running away™ from debugging and testing."""
        pass


    def get_quantization(number):
        number = round(number * 48)/48
        number -= int(number)
        return round(number*48)

# ###########################
# ###########################       
# ###########################
class PlainBeatEncoder(BeatEncoder):
    def __init__(self):
        """
        faithfully translates notes & times one by one. 

        example output: 
        <0/48> <note 1,2,3,4: tap,tap,tap,tap>
        <24/48> <note 1: tap>
        <32/48> <note 2: tap>
        <40/48> <note 3: tap>

        Inspired by MT3(J. Gardner et al.).
        """
        self.voca = OneThirtyVocabulary()

    def segmentation_policy(self, chart, current_index):
        return current_index+1
    
    def translation_policy(self, chart, current_index, next_index):
        """
        translates >>one note<<.
        chart: sequence of notes from ddc-style-jsons
        current_index: we are going to translate chart[current_index] into a sequence of tokens.
        next_index: this is here just because of the interface.

        ! invariant
        - chart[current_index] is in the "range of consideration", which is one beat in our case.
        - For all songs in the dataset, the bar boundary coincides with the beat boundary.
        """
        assert next_index == current_index + 1 
        ret = []

        the_note = chart[current_index]
        this_note_bar_beat = the_note[0]

        ret.append(self.voca.time_token[the_note[0][2] % 48]) # invariant 2
        
        encoded_notes = the_note[3]
        encoded_notes = encoded_notes.replace('3','1')
        ret.append(self.voca.chart_token[encoded_notes])

        return ret

    def encode(self, consecutive_notes):
        """
        translates >>one bar<<.
        consecutive notes: all notes in this bar.

        ! invariant
        - consecutive_notes is the entirety of notes in the "range of consideration"; one beat in our case.
        - For all songs in the dataset, the bar boundary coincides with the beat boundary.
        """
        ret = []
        length = len(consecutive_notes)
        current = 0
        while current < length:
            nxt = self.segmentation_policy(consecutive_notes, current)
            ret.extend( self.translation_policy(consecutive_notes, current, nxt) ) 
            current = nxt
        return ret

    def basethree(self, num):
        ret = [(num//27),(num//9)%3,(num//3)%3,num%3]
        ret = [str(x) for x in ret]
        return "".join(ret)

    def pretty_print(self,encoded_tokens):
        for token in encoded_tokens:
            if token == self.voca.bar_token:
                print("\n<bar>\t", end="")
            elif token == self.voca.beat_token:
                print("\n<beat>\t",end="")
            elif token in self.voca.time_token.keys():
                print(f"\n<{token:2d}/48>\t",end="")
            elif token in self.voca.chart_token.values():
                print(f"<Chart {self.basethree(token - 48)}>\t",end="")

class NBeatEncoder(): # deprecated: will note use.
    def __init__(self, how_many_beats=4, beat_encoder="plain"):
        self.how_many_beats=how_many_beats
        self.voca = SixtyTwoVocabulary()
        if beat_encoder == "plain":
            self.one_beat_encoder=PlainBeatEncoder()

    def filter_by_beat(self, consecutive_notes, beat):
        return [e for e in consecutive_notes if int(round(48 * e[1])/48) == beat ]

    def encode(self, consecutive_notes, starting_beat):
        """
        translates "how_many_beats" beats.

        ! invariant
        - consecutive_notes is the entirety of notes in the "range of consideration"; how_many_beat beats in our case.
        - For all songs in the dataset, the bar boundary coincides with the beat boundary.
        """
        ret = []
        for beat in range(starting_beat, starting_beat+self.how_many_beats):
            ret.append(self.voca.beat_token)
            ret.extend(self.one_beat_encoder.encode(self.filter_by_beat(consecutive_notes, beat)))
        return ret
    
    def pretty_print(self,encoded_tokens):
        return self.one_beat_encoder.pretty_print(encoded_tokens)

class TwoTwoEncoder():
    def __init__(self, beat_encoder="plain"):
        self.voca = OneSeventyEightVocabulary()
        self.one_beat_encoder = PlainBeatEncoder()

    def filter_by_beat(self, consecutive_notes, beat):
        return [e for e in consecutive_notes if int(round(48 * e[1])/48) == beat ]
    
    def encode(self, consecutive_notes, starting_beat):
        """
        translates 2+2 beats.

        ! invariant
        - consecutive_notes is the entirety of notes in the "range of consideration"; how_many_beat beats in our case.
        - For all songs in the dataset, the bar boundary coincides with the beat boundary.
        """
        ret = [self.voca.beat_token]
        for i in range(4):
            beat = starting_beat + i
            encoded_beat = self.one_beat_encoder.encode(self.filter_by_beat(consecutive_notes, beat))
            # translation: 130 -> 178
            if i%2==0:
                encoded_beat = [e if e<48 else e+48 for e in encoded_beat]
            else:
                encoded_beat = [e+48 for e in encoded_beat]
            ret.extend(encoded_beat)
            if i==1:
                ret.append(self.voca.beat_token)
        return ret

    def basethree(self, num):
        ret = [(num//27),(num//9)%3,(num//3)%3,num%3]
        ret = [str(x) for x in ret]
        return "".join(ret)

    def pretty_print(self, encoded_tokens):
        for token in encoded_tokens:
            if token == self.voca.bar_token:
                print("\n<bar>\t", end="")
            elif token == self.voca.beat_token:
                print("\n<beat>\t",end="")
            elif token in self.voca.time_token.keys():
                print(f"\n<{token:2d}/96>\t",end="")
            elif token in self.voca.chart_token.values():
                print(f"<Chart {self.basethree(token - 96)}>\t",end="")

class TwoTwoEncoderTimingOnly(TwoTwoEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def encode(self, consecutive_notes, starting_beat):
        """
        translates 2+2 beats.

        ! invariant
        - consecutive_notes is the entirety of notes in the "range of consideration"; how_many_beat beats in our case.
        - For all songs in the dataset, the bar boundary coincides with the beat boundary.
        """
        ret = [self.voca.beat_token]
        for i in range(4):
            beat = starting_beat + i
            encoded_beat = self.one_beat_encoder.encode(self.filter_by_beat(consecutive_notes, beat))
            # translation: 130 -> 178
            if i%2==0:
                encoded_beat = [e for e in encoded_beat if e<48 ]
            else:
                encoded_beat = [e+48 for e in encoded_beat if e<48]
            ret.extend(encoded_beat)
            if i==1:
                ret.append(self.voca.beat_token)
        return ret


class TwoTwoEncoderFrac():
    def __init__(self):
        self.voca = OneSeventyEightVocabulary()
    
    def get_quantization(self, num):
        try:
            assert num<2 and num>=0
        except AssertionError:
            print(num)
            print(num)
        return round(48*num)
    
    def encode(self, consecutive_notes, starting_beat):
        """
        translates 2+2 beats. segment start may not align with beat start.
        """
        note_on_latter_flag = False 
        ret = [self.voca.beat_token]
        starting_beat = round_to_nearest_48th(starting_beat)
        for note in consecutive_notes:
            # determine token.
            note[1] # beat
            if round_to_nearest_48th(note[1] - starting_beat) >= 2.:
                note_on_latter_flag = True
                ret.append(self.voca.beat_token)
                starting_beat += 2
            token_to_add = self.get_quantization( note[1] - starting_beat )
            if token_to_add < 96 and token_to_add >= 0:
                ret.append( token_to_add )
                ret.append( 96 + int(note[3].replace('3','1'),3) )

        if not note_on_latter_flag:
            ret.append(self.voca.beat_token)
        
        return ret


    def basethree(self, num):
        ret = [(num//27),(num//9)%3,(num//3)%3,num%3]
        ret = [str(x) for x in ret]
        return "".join(ret)
    
    def pretty_print(self, encoded_tokens):
        for token in encoded_tokens:
            if token == self.voca.bar_token:
                print("\n<bar>\t", end="")
            elif token == self.voca.beat_token:
                print("\n<beat>\t",end="")
            elif token in self.voca.time_token.keys():
                print(f"\n<{token:2d}/96>\t",end="")
            elif token in self.voca.chart_token.values():
                print(f"<Chart {self.basethree(token - 96)}>\t",end="")

class WholeSongBeatEncoder():
    def __init__(self, beat_encoder="plain"):
        self.voca = OneSeventyEightVocabulary()
        self.one_beat_encoder = PlainBeatEncoder()

    def filter_by_beat(self, consecutive_notes, beat):
        return [e for e in consecutive_notes if int(round(48 * e[1])/48) == beat ]

    def basethree(self, num):
        ret = [(num//27),(num//9)%3,(num//3)%3,num%3]
        ret = [str(x) for x in ret]
        return "".join(ret)
    
    def encode(self, consecutive_notes):
        """
        translates whole song.

        ! invariant
        - consecutive_notes is the entirety of notes in this song.
        - For all songs in the dataset, the bar boundary coincides with the beat boundary.
        """
        ret = [self.voca.beat_token]
        length_in_beats = max([int(beat) for _,beat,_,_ in consecutive_notes])
        ret = []
        for beat in range(length_in_beats):
            encoded_beat = self.one_beat_encoder.encode(self.filter_by_beat(consecutive_notes, beat))
            if beat%2==0:
                encoded_beat = [e if e<48 else e+48 for e in encoded_beat]
            else:
                encoded_beat = [e+48 for e in encoded_beat]
            ret.extend(encoded_beat)
            if beat%2==1:
                ret.append(self.voca.beat_token)
        return ret

    def pretty_print(self, encoded_tokens):
        for token in encoded_tokens:
            if token == self.voca.bar_token:
                print("\n<bar>\t", end="")
            elif token == self.voca.beat_token:
                print("\n<beat>\t",end="")
            elif token in self.voca.time_token.keys():
                print(f"\n<{token:2d}/96>\t",end="")
            elif token in self.voca.chart_token.values():
                print(f"<Chart {self.basethree(token - 96)}>\t",end="")

if __name__=="__main__":
    import sys
    import json
    
    "OSUFOLDER/153199/SHK - Couple Breaking (Sky_Demon) [MX].osu.json"
    "We will be looking at bar 3"

    with open(sys.argv[1],'r') as file:
        d = json.loads(file.read())
    consecutive_notes = d['charts'][0]['notes'][4:13]
    pce = TwoTwoEncoderFrac()
    pce.pretty_print(pce.encode(consecutive_notes, starting_beat=12.45))