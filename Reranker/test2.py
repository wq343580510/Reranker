import numpy as np
import theano
import theano.tensor as T
W_out = np.ones([3], dtype=theano.config.floatX)
b_out = np.ones([1], dtype=theano.config.floatX)


def compute_one_tree(emb):
    score = T.dot(W_out, emb) + b_out
    return score


tree_states = T.dmatrix('treestate')
tree = T.imatrix('tree')
num_nodes = tree.shape[0]  # num internal nodes
num_leaves = tree_states.shape[0] - num_nodes
internal_nodes = tree_states[num_leaves:]
scores, _ = theano.scan(
        fn=compute_one_tree,
        outputs_info=None,
        sequences=[internal_nodes],
    )

fns = theano.function(inputs=[tree_states,tree], outputs=T.sum(scores))

# tree = [[0,1,2,3,-1,-1,4],
#         [4,5,-1,-1,-1,-1,6],
#         [6,-1,-1,-1,-1,-1,7]
#         ]
tree = [[0,-1,-1,-1,-1,-1,4],[1,-1,-1,-1,-1,-1,4]]
tree_states = [[0.1,0.1,0.1],[0.2,0.2,0.2],
        [0.3,0.3,0.3],[0.4,0.4,0.4],
        [0.5,0.5,0.5],[0.6,0.6,0.6],[0.7,0.7,0.7],[0.8,0.8,0.8]]
score = fns(tree_states,tree)
print score


# def create_output_fn2(self):
#     self.W_out = theano.shared(self.init_matrix([self.hidden_dim]))
#     self.b_out = theano.shared(self.init_vector([1]))
#     self.params.extend([self.W_out, self.b_out])
#
#     def compute_one_tree(emb):
#         score = T.dot(self.W_out, emb) + self.b_out
#         return score
#
#     def fn(tree_states, tree):
#         num_nodes = tree.shape[0]  # num internal nodes
#         num_leaves = tree_states.shape[0] - num_nodes
#         internal_nodes = tree_states[num_leaves:]
#         scores, _ = theano.scan(
#             fn=compute_one_tree,
#             outputs_info=None,
#             sequences=[internal_nodes],
#         )
#         return T.sum(scores)
#
#     return fn
