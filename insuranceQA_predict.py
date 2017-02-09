#encoding=utf-8
import tensorflow as tf
import hparams 
tf.logging.set_verbosity(20)
import numpy as np
import train as train
import pickle 
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]=""
vocab = {} 


vocab_insurance = pickle.load(open('./data/vocab','r'))
test_rows = pickle.load(open('./data/test_rows', 'r'))


def vocab_processor(vocab, sent, N):
	try:
		sent = sent.split(' ')
		E = [] 
		for wd in sent:
			try:
				E.append(vocab[wd])
			except:
				continue
		length = len(E)
		if len(E) <= N:
			E = E + [len(vocab) + 1]*(N - len(E))
		else:
			E = E[-N:]
			length = N 
		return E, length
	except:
		return None, None 

def get_features(rows):
	vocab = pickle.load(open('./data/vocab','r'))
	ques_list, ans_list = [], [] 
	ques_len_list, ans_len_list = [], []
	ans_f_list, ans_f_len_list = [], []

	for ques, ans, ans_f in rows: 
		ques_encode, ques_len = vocab_processor(vocab, ques, 64)
		ans_encode, ans_len = vocab_processor(vocab, ans, 192)
		ans_f_encode, ans_f_len = vocab_processor(vocab, ans_f, 192)

		ques_list.append(ques_encode)
		ans_list.append(ans_encode)
		ques_len_list.append(ques_len)
		ans_len_list.append(ans_len)
		ans_f_list.append(ans_f_encode)
		ans_f_len_list.append(ans_f_len)

	features = {
	'ques':tf.convert_to_tensor(ques_list, dtype=tf.int64),
	'ques_len':tf.convert_to_tensor(ques_len_list, dtype=tf.int64),
	'ans':tf.convert_to_tensor(ans_list, dtype=tf.int64),
	'ans_len':tf.convert_to_tensor(ans_len_list,dtype=tf.int64),
	'ans_f':tf.convert_to_tensor(ans_f_list, dtype=tf.int64),
	'ans_f_len':tf.convert_to_tensor(ans_f_len_list, dtype=tf.int64)
	}
	return features, None

np.random.shuffle(test_rows)
estimator = tf.contrib.learn.Estimator(model_fn=train.model_fn, model_dir=hparams.model_dir)
pred = estimator.predict(input_fn = lambda: get_features(test_rows[:2000]))
print pred 
print sum(pred > 0)
print sum(pred == 0)
print sum(pred < 0)
