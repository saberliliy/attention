#coding:utf-8
import  numpy as np
import tensorflow as tf



np.random.seed(1)
# encoder hidden state [batch_size,time_step,encoder_hidden_state]
encoder_hidden=np.random.random((2,3,4))
np.random.seed(2)
# decoder hidden state [batch_size,decoder_hidden_state]
decoder_hidden=np.random.random((2,6))
u=np.random.random((6,1)) #[decoder_hidden_state,1]
w1=np.random.random((6,6))
w2=np.random.random((4,6))#[encoder_hidden_state,decoder_hidden_state]
b=np.random.random((6))
vs=[]
for  time in  range(3):
    vi = tf.matmul(tf.tanh(tf.add(tf.add(tf.matmul(decoder_hidden, w1),tf.matmul(encoder_hidden[:, time, :], w2)), b)), u)
    # 维度解释：
    #A=tf.matmul(encoder_hidden[:, time, :], w2) [ batch_size,decoder_hidden_state]
    #B=tf.matmul(decoder_hidden, w1)  [ batch_size,decoder_hidden_state]
    #C=tf.add(A,B)   [ batch_size,decoder_hidden_state]
    #D=tf.add(C,b)   [batch_size,decoder_hidden_state]
    #E=tf,matmul(D,u) [batch_size,1]
    vs.append(vi)
attention_vs = tf.concat(vs, axis=1) #[batch_size,time_step]
prob_p = tf.nn.softmax(attention_vs)
mt = tf.add_n([prob_p[:,i:i+1]*encoder_hidden[:, i, :] for i in range(3)])
with tf.Session() as sess:
    print(sess.run(prob_p))
    print(sess.run(mt))
