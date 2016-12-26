import os

import data_util
import dependency_model
from eval import eval as eval_tool

DIR = 'd:\\MacShare\\data2\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
OUTPUT_MODEL = 'model.pkl'
OUTPUT_DICT = 'dict.pkl'


def test_model():
    vocab = data_util.load_model(os.path.join(DIR,OUTPUT_DICT))
    data, max_degree, _ = data_util.get_data(os.path.join(DIR,TEST+'.kbest'),os.path.join(DIR,TEST+'.gold'),vocab)
    print 'build model'
    model = dependency_model.get_model(vocab.size(), max_degree)
    print 'load params'
    model.set_parmas(os.path.join(DIR,OUTPUT_MODEL))
    print 'addbase'
    evaluate_dataset(model,data,True)
    print 'withoutbase'
    evaluate_dataset(model,data,False)


def evaluate_baseline(data):
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        for line in inst.lines[len(inst.kbest)-1]:
            pred_trees.append(line)
        pred_trees.append('\n')
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')
    print 'baseline: %.4f' % (eval_tool.evaluate(pred_trees,gold_trees)[0])


def evaluate_oracle_worst(data):
    oracle_trees = []
    worst_trees = []
    gold_trees = []
    pred_trees = []
    for i, inst in enumerate(data):
        max = 0
        maxid = 0
        min = 1
        minid = 0
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')

        for line in inst.lines[len(inst.kbest)-1]:
            pred_trees.append(line)
        pred_trees.append('\n')

        i = 0
        for list in inst.lines:
            temp = []
            for line in list:
                temp.append(line)
            temp.append('\n')
            res = eval_tool.evaluate(temp, inst.gold_lines)[0]
            if res > max :
                max = res
                maxid = i
            if res < min :
                min = res
                minid = i
            i += 1
        for line in inst.lines[maxid]:
            oracle_trees.append(line)
        oracle_trees.append('\n')
        for line in inst.lines[minid]:
            worst_trees.append(line)
        worst_trees.append('\n')
    print 'f1score: %.4f'  % (eval_tool.evaluate(pred_trees, gold_trees)[0])
    print 'oracle: %.4f'  % (eval_tool.evaluate(oracle_trees, gold_trees)[0])
    print 'worst: %.4f'  % (eval_tool.evaluate(worst_trees, gold_trees)[0])


def evaluate_dataset(model, data , addbase):
    pred_trees = []
    gold_trees = []
    for i, inst in enumerate(data):
        pred_scores = [model.predict(tree) for tree in inst.kbest]
        data_util.normalize(pred_scores)
        if addbase:
            data_util.normalize(inst.scores)
            scores = [p_s + b_s for p_s, b_s in zip(pred_scores, inst.scores)]
        else:
            scores = pred_scores
        max_id = scores.index(max(scores))
        for line in inst.lines[max_id]:
            pred_trees.append(line)
        pred_trees.append('\n')
        for line in inst.gold_lines:
            gold_trees.append(line)
        gold_trees.append('\n')
    res = eval_tool.evaluate(pred_trees,gold_trees)
    print 'f1score: %.4f' % (res[0])
    return res
if __name__ == '__main__':
    test_model()
    test_data , _ , _ = data_util.get_data(os.path.join(DIR,TEST+'.kbest'),os.path.join(DIR,TEST+'.gold'))
    evaluate_oracle_worst(test_data)
