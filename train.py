import tensorflow as tf 
import hparams
tf.logging.set_verbosity(20)
import numpy as np
import os, sys 


def model_fn(features, targets, mode):
	ques, ques_len = features['ques'], features['ques_len']
	ans, ans_len = features['ans'], features['ans_len']
	ans_f, ans_f_len = features['ans_f'], features['ans_f_len']

	batch_size = int(ques_len.get_shape()[0])	
	ques_len = tf.reshape(ques_len, [batch_size])
	ans_len = tf.reshape(ans_len, [batch_size])
	ans_f_len = tf.reshape(ans_f_len, [batch_size])

	probs, loss = model_imp(ques, ques_len, ans, ans_len, ans_f, ans_f_len, batch_size)

	if mode == tf.contrib.learn.ModeKeys.INFER:
		return probs, 0.0, None
		
	train_op = create_train_op(loss)
	return probs, loss, train_op


def model_imp(ques, ques_len, ans, ans_len, ans_f, ans_f_len, batch_size):

	w_embed = tf.get_variable('w_embed', shape = [hparams.vocab_size, hparams.embedding_dim], initializer = tf.random_uniform_initializer(-1.0, 1.0))

	# 1.2 --- rnn for question ---
	ques = tf.nn.embedding_lookup(w_embed, ques, name = 'ques')

	with tf.variable_scope('rnn_ques') as vs_ques:
		cell = tf.nn.rnn_cell.LSTMCell(hparams.rnn_dim,forget_bias = 2.0,use_peepholes=True,state_is_tuple=True)
		cell_r = tf.nn.rnn_cell.LSTMCell(hparams.rnn_dim,forget_bias = 2.0,use_peepholes=True,state_is_tuple=True)
		output_ques, state_ques = tf.nn.bidirectional_dynamic_rnn(cell, cell_r, ques, sequence_length = ques_len, dtype = tf.float32)
	ques_output = tf.reduce_max(tf.concat(2, [output_ques[0], output_ques[1]]), 1)
	M = tf.get_variable('M', shape = [hparams.rnn_dim*2, hparams.rnn_dim*2], initializer = tf.random_uniform_initializer(-1.0, 1.0))
	ques_output = tf.matmul(ques_output, M)

	# 1.3 --- rnn for ans ---
	ans = tf.nn.embedding_lookup(w_embed, ans, name = 'ans')
	ans_f = tf.nn.embedding_lookup(w_embed, ans_f, name = 'ans_f')
	with tf.variable_scope('rnn_ques', reuse=True) as vs_ans:	
		output, state = tf.nn.bidirectional_dynamic_rnn(cell, cell_r, ans, sequence_length = ans_len,dtype = tf.float32)
	ans_output = tf.reduce_max(tf.concat(2, [output[0], output[1]]), 1)


	with tf.variable_scope('rnn_ques', reuse=True) as vs_ans_f:
		output_f, state_f = tf.nn.bidirectional_dynamic_rnn(cell, cell_r, ans_f, sequence_length = ans_f_len, dtype=tf.float32)
	ans_output_f = tf.reduce_max(tf.concat(2, [output_f[0], output_f[1]]), 1)

	# 1.4 -----------------	the prediction part ---------------------------

	ques_output = tf.nn.l2_normalize(ques_output, 1)
	ans_output = tf.nn.l2_normalize(ans_output, 1)
	ans_output_f = tf.nn.l2_normalize(ans_output_f, 1)

	simi = tf.reduce_sum(tf.mul(ques_output, ans_output), 1)
	simi_f = tf.reduce_sum(tf.mul(ques_output, ans_output_f), 1)

	loss = tf.maximum(0.0, 0.05 - simi + simi_f)

	loss_ = tf.reduce_mean(loss)
	return simi - simi_f, loss_

def create_train_op(loss):
	train_op = tf.contrib.layers.optimize_loss(loss = loss, 
		global_step = tf.contrib.framework.get_global_step(), 
		learning_rate = hparams.learning_rate, 
		clip_gradients = 10.0, 
		optimizer = hparams.optimizer)
	return train_op


def input_fn():
	# first define the features
	tf_col = tf.contrib.layers.real_valued_column
	f_col = [] 
	f_col.append(tf_col(column_name='ques', dimension=64, dtype=tf.int64))
	f_col.append(tf_col(column_name='ques_len', dimension=1, dtype=tf.int64))
	f_col.append(tf_col(column_name='ans', dimension=192, dtype=tf.int64))
	f_col.append(tf_col(column_name='ans_len', dimension=1, dtype=tf.int64))
	f_col.append(tf_col(column_name='ans_f', dimension=192, dtype=tf.int64))
	f_col.append(tf_col(column_name='ans_f_len', dimension=1, dtype=tf.int64))


	features = tf.contrib.layers.create_feature_spec_for_parsing(set(f_col))
	feature_map = tf.contrib.learn.io.read_batch_features(file_pattern=hparams.input_files,batch_size=hparams.batch_size,features=features,reader=tf.TFRecordReader,randomize_input=True,num_epochs=hparams.num_epochs,queue_capacity=1000,name="read_batch_features")
	target = None
	return feature_map, target

	
def main(unused_argv):
	estimator = tf.contrib.learn.Estimator(model_fn = model_fn,
	 model_dir = hparams.model_dir,
	 config=tf.contrib.learn.RunConfig(save_summary_steps=10, 
	 	num_cores=7, 
	 	save_checkpoints_secs=120))
	#monitor = tf.contrib.learn.monitors.ValidationMonitor(input_fn =lambda: input_fn('test'), every_n_steps=200,eval_steps =10)
	# steps is the number of steps for which to train the model. 
	estimator.fit(input_fn = input_fn, steps = None, monitors = None)

if len(sys.argv) >= 2:
	if sys.argv[1] == 'train':
		tf.app.run()
else:
	print 'infer'