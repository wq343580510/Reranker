import pickle
import theano
from theano import tensor as T
import data_util
import tree_lstm
import tree_rnn
import data_reader
FINE_GRAINED = False
DEPENDENCY = False
SEED = 88

NUM_EPOCHS = 20
LEARNING_RATE = 0.01

EMB_DIM = 300
HIDDEN_DIM = 100
OUTPUT_DIM = 3



class DependencyModel(tree_lstm.ChildSumTreeLSTM):
    def set_parmas(self,input_file):
        pkl_file = open(input_file, 'rb')
        self.embeddings.set_value(pickle.load(pkl_file))
        self.W_i.set_value(pickle.load(pkl_file))
        self.U_i.set_value(pickle.load(pkl_file))
        self.b_i.set_value(pickle.load(pkl_file))
        self.W_f.set_value(pickle.load(pkl_file))
        self.U_f.set_value(pickle.load(pkl_file))
        self.b_f.set_value(pickle.load(pkl_file))
        self.W_o.set_value(pickle.load(pkl_file))
        self.U_o.set_value(pickle.load(pkl_file))
        self.b_o.set_value(pickle.load(pkl_file))
        self.W_u.set_value(pickle.load(pkl_file))
        self.U_u.set_value(pickle.load(pkl_file))
        self.b_u.set_value(pickle.load(pkl_file))
        self.W_out.set_value(pickle.load(pkl_file))
        self.b_out.set_value(pickle.load(pkl_file))
        pkl_file.close()


    def create_output_fn(self):
        self.W_out = theano.shared(self.init_matrix([self.hidden_dim]))
        self.b_out = theano.shared(self.init_matrix([1]))
        self.params.extend([self.W_out, self.b_out])

        def fn(final_state):
            return T.dot(self.W_out, final_state) + self.b_out
        return fn

    def train_step(self, kbest_tree, gold_root):
        scores = [self.predict(tree) for tree in kbest_tree if tree.size == gold_root.size]
        max_id = scores.index(max(scores))
        pred_root = kbest_tree[max_id]
        if pred_root.size != gold_root.size:
            return 0
        gold_score = self.predict(gold_root)
        pred_score = scores[max_id]
        loss = gold_score - pred_score
        if loss < 0:
            self.train_margin(gold_root, pred_root)
        return loss

    def train_step_withbase(self, kbest_tree, gold_root, base_scores):
        pred_scores = []
        scores = []
        i = 0
        for tree in kbest_tree:
            if tree.size == gold_root.size:
                pred_scores.append(self.predict(tree))
                scores.append(base_scores[i])
            i += 1
        old_scores = pred_scores[:]
        data_util.normalize(pred_scores)
        scores = [p_s + b_s for p_s,b_s in zip(pred_scores,base_scores)]
        max_id = scores.index(max(scores))
        pred_root = kbest_tree[max_id]
        if pred_root.size != gold_root.size:
            return 0
        gold_score = self.predict(gold_root)
        pred_score = old_scores[max_id]
        loss = gold_score-pred_score
        if loss < 0:
            self.train_margin(gold_root,pred_root)
        return loss

    def loss_fn(self, gold_y, pred_y):
        return T.sum(pred_y-gold_y)

def get_model(num_emb, max_degree):
    return DependencyModel(
        num_emb, EMB_DIM, HIDDEN_DIM, OUTPUT_DIM,
        degree=max_degree, learning_rate=LEARNING_RATE,
        trainable_embeddings=True,
        labels_on_nonroot_nodes=False,
        irregular_tree=True)

