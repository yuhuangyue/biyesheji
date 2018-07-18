#coding=utf-8

import pandas as pd
import numpy as np
import tensorflow as tf

rnn_unit=128       #隐层数量
input_size=250
output_size=1
n_classes =2
lr=0.0005         #学习率
lambda_loss_amount = 0.0015
#——————————————————导入数据——————————————————————
data_len = 907
f='C:\\-----YHY-----\\--SCHOOL--\\BJFU\\Graduation_Project\\LSTM\\sentiment-network\\clip-feature-onlyframes\\features-250.xlsx'
df=pd.read_excel(f)
data=df.iloc[:,0:252].values
data = data[0:data_len]

# 归一化处理
data[:,0:input_size] -= np.mean(data[:,0:input_size], axis = 0) # zero-center
data[:,0:input_size] /= (np.std(data[:,0:input_size], axis = 0) + 1e-5) # normalize


#获取训练集
def get_train_data(batch_size=50,time_step=10,train_begin=0,train_end=data_len-30):

    batch_index=[]
    data_train = data[train_begin:train_end,0:input_size]
    data_train_y = data[train_begin:train_end, input_size:input_size+n_classes]

    train_x,train_y=[],[]   #训练集

    for i in range(len(data_train)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=data_train[i:i+time_step,:250]
       y=data_train_y[i:i+time_step,0:2]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(data_train)-time_step))


    return batch_index,train_x,train_y  # x是数据   y是标签
  
# 获取验证集
def get_validation_data(batch_size=50, time_step=10, validation_begin=data_len - 15):
    validation_test = data[validation_begin:, 0:input_size]
    validation_test_y = data[validation_begin:, input_size:input_size+n_classes]
    batch_index = []
    size = (len(validation_test) + time_step - 1) // time_step  # 有size个sample
    val_x, val_y = [], []

    for i in range(size - 1):
        if i % batch_size == 0:
            batch_index.append(i)
        x = validation_test[i * time_step:(i + 1) * time_step, :250]
        y = validation_test_y[i * time_step:(i + 1) * time_step, 0:2]
        val_x.append(x.tolist())
        val_y.append(y.tolist())
    batch_index.append((len(validation_test) - time_step))

    return batch_index,val_x, val_y  # x是数据   y是标签


#获取测试集
def get_test_data(time_step=10,test_begin=data_len-30,test_end=data_len-15):
    data_test=data[test_begin:test_end,0:input_size]
    data_test_y = data[test_begin:test_end, input_size:input_size+n_classes]

    size=(len(data_test)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    
    for i in range(size-1):
       x = data_test[i*time_step:(i+1)*time_step,:250]
       y = data_test_y[i*time_step:(i+1)*time_step,0:2]
       test_x.append(x.tolist())
       test_y.append(y.tolist())


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

    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, forget_bias=1.0, state_is_tuple=True)
    drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([drop] * 1)
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

def train_lstm(batch_size=50,time_step=10,train_begin=1,train_end=data_len):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,n_classes])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end-30)
    batch_index_val,val_x, val_y = get_validation_data(time_step=10, validation_begin=data_len-15)
    
    with tf.variable_scope("sec_lstm"):
        pred,_=lstm(X)
    Y_ = tf.reshape(Y, [-1, n_classes])

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )  # L2 loss prevents this overkill neural network to overfit the data
    for ele1 in tf.trainable_variables():
        print(ele1.name)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=pred)) + l2  # Softmax loss
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss_scalar = tf.summary.scalar('loss', loss)
    acc_scalar = tf.summary.scalar('accuracy', accuracy)
    l2_scalar = tf.summary.scalar('L2', l2)


    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter('logs', sess.graph)  # 将训练日志写入到logs文件夹下

        for i in range(30):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index)-1):

                _,loss_,acc,l2_=sess.run([train_op,loss_scalar,acc_scalar,l2_scalar],
                                 feed_dict={
                                     X:train_x[batch_index[step]:batch_index[step+1]],
                                     Y:train_y[batch_index[step]:batch_index[step+1]],
                                     keep_prob: 0.5})
                _,loss_2,acc2=sess.run([train_op,loss,accuracy],
                                 feed_dict={
                                     X:train_x[batch_index[step]:batch_index[step+1]],
                                     Y:train_y[batch_index[step]:batch_index[step+1]],
                                     keep_prob: 0.5})
            print("Number of iterations:", i+100, " train loss:", loss_2, "accuracy:", acc2)
            writer.add_summary(loss_, global_step=i)  # 写入文件
            writer.add_summary(acc, global_step=i)  # 写入文件
            writer.add_summary(l2_, global_step=i)  # 写入文件

            for step in range(len(batch_index_val)-1):
                _, loss_val, acc_val = sess.run([train_op, loss, accuracy],
                                         feed_dict={
                                             X: val_x[batch_index_val[step]:batch_index_val[step+1]],
                                             Y: val_y[batch_index_val[step]:batch_index_val[step+1]],
                                             keep_prob: 1})
            #print(" validation loss:", loss_val, "accuracy:", acc_val)

        print("model_save: ",saver.save(sess,'model_lstm\\modle_temp.ckpt'))

        print("The train has finished")
train_lstm()



#————————————————预测模型————————————————————
def prediction(time_step=10):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,n_classes])
    test_x,test_y=get_test_data(time_step)
    test_y = np.array(test_y)
    test_y_ = tf.reshape(test_y,[-1,n_classes])
    #print ("测试集大小：",len(test_x),len(test_y))
    
    
    with tf.variable_scope("sec_lstm",reuse=True):
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_lstm')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)): #我这里删掉了一个-1
          prob,y=sess.run([pred,test_y_],
                          feed_dict={X:[test_x[step]],
                                     Y:[test_y[step]],
                                     keep_prob: 1})
          predict=prob.reshape([-1,n_classes])
          test_predict.extend(predict)
    
        test_predict=np.array(test_predict) #相当于解决归一化
        print ("真实值 ",y,"预测结果 ",test_predict)

        correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(test_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        acc=sess.run([accuracy],feed_dict={X:[test_x[step]],
                                           Y:[test_y[step]],
                                           keep_prob: 1})
        print("The accuracy of this predict:",acc)


prediction()
