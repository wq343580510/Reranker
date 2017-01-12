import data_util
import os
DIR = 'd:\\MacShare\\data\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
OUTPUT_MODEL = 'model.pkl'
model = data_util.load_model(os.path.join(DIR, OUTPUT_MODEL))
print 'end'