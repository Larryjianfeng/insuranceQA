# insuranceQA
tensorflow implementation of insuranceQA lstm pairs

The ./data/train_tf.records is lost, because the file size is beyond the maximum size of github files. 
You can create it by your self.

use pickle.load(open('./data/rows.dump','r')) to load the original data, the data is extracted from the insuranceQA by https://github.com/shuzi/insuranceQA.

Be careful, this is only a basic tensorflow implementation of lstm model, no deep parameters tunning is conducted. 

If you are interested in the QA modeling with deep learning. Contact me at jy2641@columbia.edu. 
