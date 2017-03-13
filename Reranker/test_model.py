import data_util
import os
import pickle
DIR = 'd:\\MacShare\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
OUTPUT_MODEL = 'model_best.pkl'
#model = data_util.load_model(os.path.join(DIR, OUTPUT_MODEL))

def set_parmas(input_file):
    pkl_file = open(input_file, 'rb')
    a1 =pickle.load(pkl_file)
    a2 =pickle.load(pkl_file)
    a3 = pickle.load(pkl_file)
    a4 = pickle.load(pkl_file)
    a5 = pickle.load(pkl_file)
    a6 = pickle.load(pkl_file)
    a7 = pickle.load(pkl_file)
    a8 = pickle.load(pkl_file)
    a9 = pickle.load(pkl_file)
    a10 = pickle.load(pkl_file)
    a11 = pickle.load(pkl_file)
    pkl_file.close()
set_parmas(os.path.join(DIR, OUTPUT_MODEL))
print 'end'