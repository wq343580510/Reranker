import pickle
import math



class instance(object):
    def __init__(self,kbest,scores,gold,lines,gold_lines):
        self.kbest = kbest
        self.scores = scores
        self.gold = gold
        self.gold_lines = gold_lines
        self.lines = lines


def normalize(list):
    sum = 0
    max_score = max(list)
    for i in range(len(list)):
        list[i] = math.pow(1.1, list[i]-max_score)
        sum += list[i]

    for i in range(len(list)):
        list[i] = math.log(list[i]/sum, 1.1)



def save_model(model,output_file):
    output = open(output_file, 'wb')
    for shared_value in model.params:
        pickle.dump(shared_value.get_value(),output, protocol=2)
    output.close()


def save_dict(vocab,degree,output_file):
    output = open(output_file, 'wb')
    pickle.dump(degree,output, protocol=2)
    pickle.dump(vocab,output, protocol=2)
    output.close()

def load_dict(input_file):
    pkl_file = open(input_file, 'rb')
    degree = pickle.load(pkl_file)
    dict = pickle.load(pkl_file)
    pkl_file.close()
    return degree,dict

def load_model(input_file):
    pkl_file = open(input_file, 'rb')
    params = pickle.load(pkl_file)
    pkl_file.close()
    return params