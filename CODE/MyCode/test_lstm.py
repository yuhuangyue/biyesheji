#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn 
import xlwt

rnn_unit=128       #隐层数量
input_size=250
output_size=1
n_classes =2
lr=0.001         #学习率
#——————————————————导入数据——————————————————————

f='C:\\Users\\56891\\Desktop\\篮球视频结果\\更新代码-1\\features\\test1_features.xls'
df=pd.read_excel(f)     #读入股票数据
data=df.iloc[:,0:(input_size + n_classes)].values  #取第3-10列
data_len = 226
data = data[0:data_len]

# 归一化处理
data[:,0:input_size] -= np.mean(data[:,0:input_size], axis = 0) # zero-center
data[:,0:input_size] /= (np.std(data[:,0:input_size], axis = 0) + 1e-5) # normalize
  

#获取测试集
def get_test_data(time_step=20,test_begin=0):

    data_test=data
    size=(len(data_test)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    
    for i in range(size-1):
       x=data_test[i*time_step:(i+1)*time_step , :input_size]
       y=data_test[i*time_step:(i+1)*time_step , input_size:input_size+n_classes]
       test_x.append(x.tolist())
       test_y.append(y.tolist())

    print ("------get testing data----")
    return test_x,test_y  # x是数据   y是标签


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

    cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit, forget_bias=1.0, state_is_tuple=True)
    drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([drop] * 1)
    print ("-------------输出cell")


    output_rnn,final_states=tf.nn.dynamic_rnn(lstm_cell, input_rnn, dtype=tf.float32)
    print ("-------------跑rnn部分")
    output=tf.reshape(output_rnn,[-1,rnn_unit]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    print ("-------------lstm训练结束！！！")
    return pred,final_states


#————————————————预测模型————————————————————
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,n_classes])
    test_x,test_y=get_test_data(time_step)
    test_y = np.array(test_y)
    test_y_ = tf.reshape(test_y,[-1,n_classes])
    print ("测试集大小：",len(test_x),len(test_y))
    
    
    with tf.variable_scope("sec_lstm",reuse=None):
        pred,_=lstm(X)



    saver=tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        #参数恢复

        module_file = tf.train.latest_checkpoint('model_lstm')
        saver.restore(sess, module_file)
        print ('model is ok')

        test_predict=[]

        for step in range(len(test_x)):
          prob,y=sess.run(
              [pred,test_y_],
              feed_dict={
                  X:[test_x[step]],
                  Y:[test_y[step]],
                  keep_prob:1})

          predict=prob.reshape([-1,n_classes])
          test_predict.extend(predict)
    
        test_predict=np.array(test_predict)

        print (test_predict)
                
        #save in excel
        book = xlwt.Workbook(encoding='utf-8',style_compression = 0)
        sheet = book.add_sheet('test',cell_overwrite_ok = True)
        list_ans = []
        for i in range(220):
            sheet.write(i,0,float(test_predict[i][0]))
            sheet.write(i,1,float(test_predict[i][1]))
            if test_predict[i][0]>test_predict[i][1]:
                list_ans.append(i)

        book.save('C:\\Users\\56891\\Desktop\\篮球视频结果\\更新代码-1\\ans\\strong\\test1.xls')







        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(test_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acc = sess.run(
            [accuracy],
            feed_dict={
                X: [test_x[step]],
                Y: [test_y[step]],
                keep_prob: 1})
        print("The accuracy of this predict:",acc)
        list_ans.append(int(float(acc[0])*100))

        #--------下面计算mAP
        result = np.hstack((test_predict,y))
        sort_result = result[np.lexsort(-result[:,::-1].T)]
        AP = 0
        pos = 1
        for ii in range(len(sort_result)):
            if sort_result[ii,0]>sort_result[ii,1] and sort_result[ii,2]==1:
                AP = pos/(ii+1) + AP
                pos = pos + 1


        # --------下面计算F-score
        a = 0
        b = 0
        a2 = 0
        b2 = 0
        for ii in range(len(sort_result)):
            if sort_result[ii,0] > sort_result[ii,1] and sort_result[ii,2]==1:
                a2 = a2 + 1
            if sort_result[ii, 0] < sort_result[ii, 1] and sort_result[ii, 3] == 1:
                b2 = b2 + 1

            if sort_result[ii, 2] == 0:
                b = b+1
            else:
                a = a+1

        precious = a2 / (a2 + b2);
        recall = a2 / a;
        fscore = (2 * precious * recall) / (precious + recall)

        print("mAP : ", AP / pos, "F-score : ",fscore)
        list_ans.append(int(AP/pos*100))
        list_ans.append(int(fscore*100))

        list_str = ','.join('%s' %id for id in list_ans)

prediction()
