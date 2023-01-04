from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer:
            # TODO: Write the body of this loop for part 4
            #print(state.stack)
            inp = self.extractor.get_input_representation(words, pos, state)
            inp = inp.reshape(-1, inp.shape[0])
            pred_transitions = self.model.predict(inp).flatten()
            sorted_pred_transition_inds = list(reversed(sorted([(i, prob) for i, prob in enumerate(pred_transitions)], key=lambda x: x[1])))
            sorted_pred_transitions = [self.output_labels[i] for (i, prob) in sorted_pred_transition_inds]

            for transition in sorted_pred_transitions:
                if transition[0] == 'shift':
                    if len(state.stack) == 0 or len(state.buffer) > 1: # stack is empty or word being shifted isn't the last word on buffer, so transition is permitted
                        state.shift()
                        break
                elif transition[0] == 'left_arc':
                    if len(state.stack) > 0 and state.stack[-1] != 0: # stack is not empty or target of left-arc isn't root, so transition is permitted
                        state.left_arc(transition[1])
                        break
                elif transition[0] == 'right_arc' and len(state.stack) > 0:
                    state.right_arc(transition[1])
                    break

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
