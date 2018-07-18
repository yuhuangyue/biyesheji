#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

rnn_unit=10       #隐层数量 （这个需要调节）
input_size=250
output_size=1
n_classes =2
lr=0.0006         #学习率
#——————————————————导入数据——————————————————————

train_validation_len = 860
train_data_len = 907

train_validation='C:\\-----YHY-----\\--SCHOOL--\\BJFU\\Graduation_Project\\LSTM\\sentiment-network\\clip-feature-onlyframes\\features-250-unsup.xlsx'
df=pd.read_excel(train_validation)
data_validation=df.iloc[:,0:input_size].values
data_validation = data_validation[0:train_validation_len]

train_data='C:\\-----YHY-----\\--SCHOOL--\\BJFU\\Graduation_Project\\LSTM\\sentiment-network\\clip-feature-onlyframes\\features-250.xlsx'
df2=pd.read_excel(train_data)
data=df2.iloc[:,0:input_size].values
data = data[0:train_data_len]

# 归一化处理
data[:,0:input_size] -= np.mean(data[:,0:input_size], axis = 0) # zero-center
data[:,0:input_size] /= (np.std(data[:,0:input_size], axis = 0) + 1e-5) # normalize
data_validation[:,0:input_size] -= np.mean(data_validation[:,0:input_size], axis = 0) # zero-center
data_validation[:,0:input_size] /= (np.std(data_validation[:,0:input_size], axis = 0) + 1e-5) # normalize

#获取训练集
def get_train_data(dataset,batch_size,time_step,train_begin,train_end):
    batch_index=[]

    data_train=data[train_begin:train_end,0:input_size]

    train_x=[]  #训练集
    for i in range(len(data_train)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=data_train[i:i+time_step,:input_size]
       train_x.append(x.tolist())

    batch_index.append((len(data_train)-time_step))

    return batch_index,train_x # x是数据

#获取验证集
def get_validation_data(dataset,batch_size,time_step,train_begin,train_end):
    batch_index=[]

    data_validation=data[train_begin:train_end,0:input_size]

    validation_x=[]  #训练集
    for i in range(len(data_validation)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=data_validation[i:i+time_step,:input_size]
       validation_x.append(x.tolist())

    batch_index.append((len(data_validation)-time_step))

    return batch_index,validation_x # x是数据


# 获取测试集
def get_test_data(time_step=10, test_begin=907-860, test_end=907):
    data = df2.iloc[:, 0:input_size+n_classes].values
    data_test = data[test_begin:test_end, 0:input_size]
    data_test_y = data[test_begin:test_end, input_size:input_size + n_classes]

    size = (len(data_test) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []

    for i in range(size - 1):
        x = data_test[i * time_step:(i + 1) * time_step, :250]
        y = data_test_y[i * time_step:(i + 1) * time_step, 0:2]
        test_x.append(x.tolist())
        test_y.append(y.tolist())

    return test_x, test_y  # x是数据   y是标签


#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,n_classes]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
       }
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    print ("-------------开始进行lstm的训练")
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, forget_bias=1.0, state_is_tuple=True)
    drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([drop] * 2)
    print ("-------------输出cell",cell)

    output_rnn,final_states=tf.nn.dynamic_rnn(lstm_cell, input_rnn, dtype=tf.float32)
    print ("-------------跑rnn部分",output_rnn)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    print ("-------------lstm训练结束！！！")
    return pred,final_states




#————————————————训练模型————————————————————

def train_lstm(batch_size,time_step):
    X = tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    X_validation = tf.placeholder(tf.float32, shape=[None, time_step, input_size])

    #训练集
    batch_index,train_x=get_train_data(data,batch_size,time_step,0,train_data_len)
    batch_index_validation, train_x_validation = get_train_data(data_validation, batch_size, time_step, 0, train_validation_len)

    #验证集
    batch_index_val,train_x_val=get_validation_data(data,batch_size,time_step,0,train_data_len)
    batch_index_validation_val, train_x_validation_val = get_validation_data(data_validation, batch_size, time_step, 0, train_validation_len)



    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)


    with tf.variable_scope("sec_lstm_validation"):
        pred_validation,_=lstm(X_validation)

    # Loss, optimizer and evaluation
    lambda_loss_amount = 0.0015
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )  # L2 loss prevents this overkill neural network to overfit the data

    # ***********先暂时这么写 看看有没有别的问题
    pred_validation_ = tf.nn.softmax(pred_validation)
    pred_ = tf.nn.softmax(pred)
    loss = tf.reduce_mean(tf.square(pred_[:,0])-tf.square(pred_validation_[:,0])) + l2  # Softmax loss
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)  # Adam Optimizer


    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            batch_len = min(len(batch_index),len(batch_index_validation))
            for step in range(batch_len-2):
                feed = {X: train_x[batch_index[step]:batch_index[step + 1]],
                        X_validation: train_x_validation[
                                      batch_index_validation[step]:
                                      batch_index_validation[step + 1]],
                        keep_prob: 0.5}

                _,loss_=sess.run([train_op,loss],feed_dict=feed)
            print("Number of iterations:",i," loss:",loss_)

            batch_len_val = min(len(batch_index_val), len(batch_index_validation_val))
            for step in range(batch_len_val-2):
                feed_val = {X: train_x_val[batch_index_val[step]:batch_index_val[step + 1]],
                        X_validation: train_x_validation_val[
                                      batch_index_validation_val[step]:
                                      batch_index_validation_val[step + 1]],
                        keep_prob: 1}

                _, loss_val = sess.run([train_op, loss], feed_dict=feed_val)
            print(" validation loss:", loss_val)
        print("model_save: ",saver.save(sess,'model_lstm\\modle.ckpt'))

        print("The train has finished")
train_lstm(50,20)


# ————————————————预测模型————————————————————
def prediction(time_step=10):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, n_classes])
    test_x, test_y = get_test_data(time_step)
    test_y = np.array(test_y)
    test_y_ = tf.reshape(test_y, [-1, n_classes])

    with tf.variable_scope("sec_lstm", reuse=True):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_lstm')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x)):  # 我这里删掉了一个-1
            prob, y = sess.run([pred, test_y_],
                               feed_dict={X: [test_x[step]],
                                          Y: [test_y[step]],
                                          keep_prob: 1})
            predict = prob.reshape([-1, n_classes])
            test_predict.extend(predict)

        test_predict = np.array(test_predict)  # 相当于解决归一化
        print("真实值 ", y, "预测结果 ", test_predict)

        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(test_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        acc = sess.run([accuracy], feed_dict={X: [test_x[step]],
                                              Y: [test_y[step]],
                                              keep_prob: 1})
        print("The accuracy of this predict:", acc)


prediction()
