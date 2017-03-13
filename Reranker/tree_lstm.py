import tree_rnn

import theano
from theano import tensor as T


class ChildSumTreeLSTM(tree_rnn.TreeRNN):
    def create_recursive_unit(self):
        self.params['W_i'] = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.params['U_i'] = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.params['b_i'] = theano.shared(self.init_vector([self.hidden_dim]))
        self.params['W_f'] = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.params['U_f'] = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.params['b_f'] = theano.shared(self.init_vector([self.hidden_dim]))
        self.params['W_o'] = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.params['U_o'] = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.params['b_o'] = theano.shared(self.init_vector([self.hidden_dim]))
        self.params['W_u'] = theano.shared(self.init_matrix([self.hidden_dim, self.emb_dim]))
        self.params['U_u'] = theano.shared(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.params['b_u'] = theano.shared(self.init_vector([self.hidden_dim]))
        self.regular['W_i'] = self.params['W_i']
        self.regular['U_i'] = self.params['U_i']
        self.regular['W_f'] = self.params['W_f']
        self.regular['U_f'] = self.params['U_f']
        self.regular['W_o'] = self.params['W_o']
        self.regular['U_o'] = self.params['U_o']
        self.regular['W_u'] = self.params['W_u']
        self.regular['U_u'] = self.params['U_u']
        # self.params.extend([
        #     self.W_i, self.U_i, self.b_i,
        #     self.W_f, self.U_f, self.b_f,
        #     self.W_o, self.U_o, self.b_o,
        #     self.W_u, self.U_u, self.b_u])

        def unit(parent_x, child_h, child_c, child_exists):
            h_tilde = T.sum(child_h, axis=0)
            i = T.nnet.sigmoid(T.dot(self.params['W_i'], parent_x) + T.dot(self.params['U_i'], h_tilde) + self.params['b_i'])
            o = T.nnet.sigmoid(T.dot(self.params['W_o'], parent_x) + T.dot(self.params['U_o'], h_tilde) + self.params['b_o'])
            u = T.tanh(T.dot(self.params['W_u'], parent_x) + T.dot(self.params['U_u'], h_tilde) + self.params['b_u'])

            f = (T.nnet.sigmoid(
                    T.dot(self.params['W_f'], parent_x).dimshuffle('x', 0) +
                    T.dot(child_h, self.params['U_f'].T) +
                    self.params['b_f'].dimshuffle('x', 0)) *
                 child_exists.dimshuffle(0, 'x'))

            c = i * u + T.sum(f * child_c, axis=0)
            h = o * T.tanh(c)
            return h, c

        return unit

    def create_leaf_unit(self):
        # 50 * hidden
        dummy = 0 * theano.shared(self.init_vector([self.degree, self.hidden_dim]))
        def unit(leaf_x):
            #parent_x, child_h, child_c, child_exists
            return self.recursive_unit(
                leaf_x, #50:emb
                dummy,  #12*200 degree*emb
                dummy,  #12*200 degree*emb
                dummy.sum(axis=1)) #200 hidden
        return unit

    def compute_tree(self, emb_x, tree,labels): #15 * 50 ,12 * 5,15
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # compute leaf hidden states
        (leaf_h, leaf_c), _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        if self.irregular_tree:
            init_node_h = T.concatenate([leaf_h, leaf_h], axis=0) #leaf node 10 10*200 -> 20 *200
            init_node_c = T.concatenate([leaf_c, leaf_c], axis=0)
        else:
            init_node_h = leaf_h
            init_node_c = leaf_c
        def composition(child_h,child_id,parent_emb,parent_id, labels):
            parent_tag = labels[parent_id]
            child_tag =  labels[child_id]
            res = T.switch(T.lt(-1,child_id), T.tanh(T.dot(T.concatenate([parent_emb,child_h]),self.compMatrix[parent_tag][child_tag])),
                           T.zeros(self.hidden_dim))
            return res
        def score_pair(child_h,child_id,parent_h,parent_id, labels):
            parent_tag = labels[parent_id]
            child_tag =  labels[child_id]
            score = T.switch(T.lt(-1,child_id), T.tanh(T.dot(T.concatenate([parent_h,child_h]),self.scoreVector[parent_tag][child_tag])),
                           T.zeros(self.hidden_dim))
            return score
        # use recurrence to compute internal node hidden states
        # !!!note that the node_info don't contain the parent_id
        # a example if node_h = [leaf0,leaf1,leaf2,leaf3,leaf4,leaf0_c,leaf1_c,leaf2_c,leaf3_c,leaf4_c]
        # leaf_c is the same as leaf just a symbol
        # num_leaves = 5 isrregular_tree = true
        # tree = [[0,1,2,-1,-1,5],[3,5,-1,-1,-1,6],[4,6,-1,-1,-1,7]] of course the 5 6 7 should be get out when compute tree
        # so in the first iter node_info = [0,1,2,-1,-1] t = 0
        # child_exists = [1,1,1,0,0]
        # offset = 5-[1,1,1,0,0]*0  node_info + offset = [5,6,7,4,4]
        # so the child_h = [leaf0_c,leaf1_c,leaf2_c,0,0] the index -1 the child_h will be zero
        # in the second iter
        # node_info = [3,5,-1,-1,-1] t = 1
        # and node_h = [leaf1,leaf2,leaf3,leaf4,leaf0_c,leaf1_c,leaf2_c,leaf3_c,leaf4_c,internal_5]
        # child_exists = [1,1,0,0,0] offset = 5-[1,1,0,0,0]*1=[4,4,5,5,5] node_info + offset = [7,9,4,4,4]
        # so the child_h [leaf3_c,internal_5,0,0,0]
        def _recurrence(cur_emb, node_info, parent_id, t, node_h, node_c, last_h,labels):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            #dimshuffle change the 1*12 to 12*1
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            z_h,_  = theano.scan(
                 fn = composition,
                 outputs_info=None,
                 sequences=[child_h,node_info],
                 non_sequences=[cur_emb,parent_id,labels]
             )
            parent_h, parent_c = self.recursive_unit(cur_emb, z_h, child_c, child_exists)
            scores, _ = theano.scan(
                fn=score_pair,
                outputs_info=None,
                sequences=[child_h, node_info],
                non_sequences=[parent_h, parent_id, labels]
            )
            score = T.sum(scores)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c,
                                    parent_c.reshape([1, self.hidden_dim])])
            return node_h[1:], node_c[1:], parent_h,score

        dummy = theano.shared(self.init_vector([self.hidden_dim]))
        (_, _, parent_h,score), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, init_node_c, dummy,None],
            sequences=[emb_x[num_leaves:], tree[:,:-1],tree[:,-1], T.arange(num_nodes)],
            non_sequences = [labels],
            n_steps=num_nodes)
        return T.concatenate([leaf_h, parent_h], axis=0) , T.sum(score)
