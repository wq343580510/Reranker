import pickle
import theano
import numpy
from theano import tensor as T
import data_util
import tree_lstm
import tree_rnn
import data_reader
FINE_GRAINED = False
DEPENDENCY = False
SEED = 88

NUM_EPOCHS = 20
LEARNING_RATE = 0.1


DECAY = 0.002
EMB_DIM = 50
HIDDEN_DIM = 200
OUTPUT_DIM = 3


def _p(pp, name1 ,name2):
    return '%s_%s-%s' % (pp, name1, name2)

class DependencyModel(tree_lstm.ChildSumTreeLSTM):
    def set_params(self,params):
        self.embeddings.set_value(params['embeddings'].get_value())
        self.scoreVector.set_value(params['scoreVector'].get_value())
        self.compMatrix.set_value(params['compMatrix'].get_value())
        self.params['W_i'].set_value(params['W_i'].get_value())
        self.params['U_i'].set_value(params['U_i'].get_value())
        self.params['b_i'].set_value(params['b_i'].get_value())
        self.params['W_f'].set_value(params['W_f'].get_value())
        self.params['U_f'].set_value(params['U_f'].get_value())
        self.params['b_f'].set_value(params['b_f'].get_value())
        self.params['W_o'].set_value(params['W_o'].get_value())
        self.params['U_o'].set_value(params['U_o'].get_value())
        self.params['b_o'].set_value(params['b_o'].get_value())
        self.params['W_u'].set_value(params['W_u'].get_value())
        self.params['U_u'].set_value(params['U_u'].get_value())
        self.params['b_u'].set_value(params['b_u'].get_value())

    # def create_output_fn(self):
    #     def fn(tree_states, tree,labels):
    #         score = T.switch(T.lt(-1, one_child),
    #                      T.dot(self.scoreVector[parent_tag][child_tag],
    #                            T.concatenate([tree_states[parent], tree_states[one_child]])),
    #                      T.zeros(1))
    #
    #     return score
    def create_output_fn(self):
        # self.W_out = theano.shared(self.init_matrix([2*self.hidden_dim]))
        # self.b_out = theano.shared(self.init_vector([1]))
        # #self.params.extend([self.W_out, self.b_out])
        # self.params['W_out'] = self.W_out
        # self.params['b_out'] = self.b_out


        def compute_one_edge(one_child, tree_states , parent,labels):
            # if not self.params.has_key(_p('W_out',parent_tag,labels[one_child])):
            #     self.params[_p('W_out',parent_tag,labels[one_child])] = theano.shared(self.init_matrix([2*self.hidden_dim]))
            #     self.params[_p('b_out',parent_tag,labels[one_child])] = theano.shared(self.init_vector([1]))
            # W_out = self.params.get(_p('W_out',parent_tag,labels[one_child]))
            # b_out = self.params.get(_p('b_out',parent_tag,labels[one_child]))
            parent_tag = labels[parent]
            child_tag =  labels[one_child]
            score = T.switch(T.lt(-1,one_child),
                             T.dot(self.scoreVector[parent_tag][child_tag], T.concatenate([tree_states[parent], tree_states[one_child]])),
                             T.zeros(1))
            return score

        def compute_one_tree(one_tree, tree_states,labels):
            children = one_tree[0:-1]
            parent_id = one_tree[-1]
            result,_ = theano.scan(
                fn=compute_one_edge,
                outputs_info=None,
                sequences=[children],
                non_sequences=[tree_states,parent_id,labels],
            )
            return T.sum(result)

        def fn(tree_states, tree,labels):
            scores ,_ = theano.scan(
                fn=compute_one_tree,
                outputs_info=None,
                sequences=[tree],
                non_sequences=[tree_states,labels],
            )
            return T.sum(scores)
        return fn

    def train_step(self, kbest_tree, gold_root):
        scores = []
        for tree in kbest_tree:
            if tree.size == gold_root.size:
                scores.append(self.predict(tree))
            else:
                scores.append(-1000)
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

    def train_step_oracle(self, inst):
        scores = []
        for tree in inst.kbest:
            if tree.size == inst.gold.size:
                scores.append(self.predict(tree))
            else:
                scores.append(-1000)
        max_id = scores.index(max(scores))
        pred_score = scores[max_id]
        pred_root = inst.kbest[max_id]
        oracle_id = inst.get_oracle_index()
        oracle_root = inst.kbest[oracle_id]
        if pred_root.size != oracle_root.size:
            return 0
        oracle_score = self.predict(oracle_root)
        loss = oracle_score-pred_score
        if loss > 0:
            self.train_margin(oracle_root,pred_root)
        return loss

    def loss_fn(self, gold_y, pred_y):
        loss = T.sum(pred_y-gold_y)
        regular = 0
        # L2 = T.sum(self.W_o ** 2)+T.sum(self.W_i ** 2)
        # L3 = (T.sum(self.W_o ** 2)+T.sum(self.W_i ** 2))
        for param in self.regular.values():
            regular += T.sum(param ** 2)
        return loss + DECAY * regular

def get_model(num_emb, num_tag ,max_degree):
    return DependencyModel(
        num_emb, num_tag,EMB_DIM, HIDDEN_DIM, OUTPUT_DIM,
        degree=max_degree, learning_rate=LEARNING_RATE,
        trainable_embeddings=True,
        labels_on_nonroot_nodes=False,
        irregular_tree=True)

