�⼸���ļ�����Ҫ��
eval_ratio oracle 
test 
test_error_tree
train_data_iter
train_iterator


data_reader.py:
	data_manager��Ҫ�ǽ����ʵ䣬������Ķȣ�tree_lstm����modelʱ���õ�����
				��ȡtrain��dev
data_util.py
	instance ��ʾһ��ѵ������
	����һЩ�����͵ĺ���

dev_reader.py ��ȡdev���ϵ��࣬����dev data��Ҫ��f1ֵ��train data����Ҫ
�����е㲻һ����

��Ҫ��΢���ӵĴ�����
tree_rnn.py����
�����ʵ����tree_rnn model��Ȼ��tree_lstm�������������չ��
�����õ�dependency_model������tree_lstm�����ࡣ
tree_lstm��NaryTreeLSTM���ùܡ�

������������reranker_train.py
Ψһ�е㸴�ӵĵط���tree_rnn.py����
tree_lstm��tree_rnn.py�����ࡣ

������tree_rnn��Դ���룬���������Ļ����ϸ���һЩ����
https://github.com/ofirnachum/tree_rnn