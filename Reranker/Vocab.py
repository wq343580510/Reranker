class Vocab(object):
    def __init__(self,filename):
        self.words = []
        self.postag = []
        self.pos2idx = {}
        self.word2idx = {}
        self.unk_index = -1
        self.unk_token = 'unk'
        with open(filename) as reader:
            data = reader.readlines()
            for line in data:
                if line.strip() != 'PTB_KBEST' and '_' in line:
                    split = line.split()
                    word = split[1]
                    if not word in self.words:
                        self.words.append(word)
                        self.word2idx[word]=len(self.words)-1
                    pos = split[3]
                    if not pos in self.postag:
                        self.postag.append(pos)
                        self.word2idx[pos]=len(self.postag)-1
        reader.close()

    def index_pos(self,pos):
        return self.pos2idx.get(pos, self.unk_index)
    def size_pos(self):
        return len(self.postag)
    def index(self,word):
        return self.word2idx.get(word, self.unk_index)
    def size(self):
        return len(self.words)


