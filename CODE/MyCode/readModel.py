# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import re
import os

model_dir='C:/Oops!yhy/Learning/Graduation_Project/inception-v3/inception-v3/inception_dec_2015/'
image='C:/Oops!yhy/Learning/Graduation_Project/inception-v3/inception-v3/cat.jpg'




def train_with_inception():
  #log_dir = 'C:/Oops!yhy/Learning/Graduation_Project/inception-v3/inception-v3/tensorboard' 
  with tf.gfile.FastGFile(os.path.join(model_dir, 'tensorflow_inception_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    inception_tensor,jpeg_data_tensor = tf.import_graph_def(graph_def,return_elements=['mixed_10/join:0','DecodeJpeg/contents:0'])


 # inception_tensor = g.get_tensor_by_name('mixed_10/join:0') # 返回给定名称的tensor
  return inception_tensor,jpeg_data_tensor


"""

#读取图片
image_data = tf.gfile.FastGFile(image, 'rb').read()



sess=tf.Session()


#创建graph
train_with_inception()


#Inception-v3模型的最后一层softmax的输出
softmax_tensor = sess.graph.get_tensor_by_name('mixed_10/join:0') # 返回给定名称的tensor
print ("这个是shape是多少",softmax_tensor)

#输入图像数据，得到softmax概率值（一个shape=(1,1008)的向量）
predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
print ("这个是shape是多少",predictions)

#(1,1008)->(1008,)
predictions = np.squeeze(predictions)

# ID --> English string label.
#node_lookup = NodeLookup()

#取出前5个概率最大的值（top-5)
top_5 = predictions.argsort()[-5:][::-1]
      
#读取标签
label_lookup_path = os.path.join(model_dir, 'imagenet_comp_graph_label_strings.txt')
proto_as_ascii_lines = tf.gfile.GFile(label_lookup_path).readlines()


for node_id in top_5:
  human_string = proto_as_ascii_lines[node_id]
  score = predictions[node_id]
  print('%s (score = %.5f)' % (human_string, score))

sess.close()

"""