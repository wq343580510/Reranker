import dev_reader
import os
import parser_test
DIR = 'd:\\MacShare\\data2\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

if __name__ == '__main__':
    test_data = dev_reader.read_dev(os.path.join(DIR, DEV + '.kbest'),
                                    os.path.join(DIR, DEV + '.gold'), None)
    parser_test.evaluate_oracle_worst(test_data)
    test_data = dev_reader.read_dev(os.path.join(DIR,TEST+'.kbest'),
                         os.path.join(DIR,TEST+'.gold'),None)
    parser_test.evaluate_oracle_worst(test_data)