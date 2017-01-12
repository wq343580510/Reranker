import theano
import numpy as np
import theano.tensor as T

# y = T.imatrix('y')
# x = T.imatrix('x')
# z = T.concatenate([x,y])
# a = z.sum(axis=1)
# f = theano.function(inputs=[x,y], outputs=a)
# zre = f([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]],[[6,7,8,9,10],[6,7,8,9,10]])
# print zre

# node_info = T.ivector('x')
# node_h = T.imatrix('h')
# child_exists = node_info > -1
# offset = 5 - child_exists * 1
# child_h = node_h[node_info] * child_exists.dimshuffle(0, 'x')
# f = theano.function(inputs=[node_info,node_h], outputs=[])
# res = f([0,1,2,-1,-1,4],[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6],[7,7,7]])
# print res

def composition(child_h, child_id, parent_emb):
    W = theano.shared(np.random.normal(size=10))
    res = T.switch(T.lt(-1, child_id),
                   T.tanh(T.dot(W, T.concatenate([parent_emb, child_h]))),
                   T.zeros(5))
    return res
child_h = T.fmatrix('ch')
node_info = T.ivector('2')
cur_emb = T.fvector('c')

z_h,_ = theano.scan(
    fn=composition,
    outputs_info=None,
    sequences=[child_h, node_info],
    non_sequences=[cur_emb]
)
f = theano.function(inputs=[child_h,node_info,cur_emb], outputs=[z_h])
child_h = np.random.normal(size=[6,5])
cur_emb = np.random.normal(size=5)
node_info = [0,1,3,-1,-1,-1]