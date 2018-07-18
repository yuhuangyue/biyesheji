# -*- coding: utf-8 -*-
import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from skimage import io, transform
import os



g1 = tf.Graph()  #建立图1，原始模型
g2 = tf.Graph()  #建立图2，后来训练的模型


dir_path = "C:\Oops!yhy\Learning\Graduation_Project\inception-v3\inception-v3\model/"
file = "C:\Oops!yhy\Learning\Graduation_Project\inception-v3\inception-v3\data\web\web/"
label_path = "labels.txt"
ans_path = "result.txt"

def ReadImg(filepath, pb_file_path1,pb_file_path2):


    with g1.as_default():

        output_graph_def = tf.GraphDef()

        with open(pb_file_path1, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:

            #init = tf.global_variables_initializer()

            #sess.run(init)
            
            input_x = g1.get_tensor_by_name("DecodeJpeg/contents:0")
            #print ("这是第一步！",input_x)
            bottleneck_tensor = g1.get_tensor_by_name("pool_3/_reshape:0")
            bottl_values = []
            for filename in os.listdir(filepath):              #listdir的参数是文件夹的路径
                print( "我看看文件夹里面的东西？",filename  )                              #此时的filename是文件夹中文件的名称
                if filename[0]!='.':
                    filename = file + filename
                    bottl = recognize(sess , filename )
                    bottl_values.append(bottl)
                    #recognize2(sess2,bottl_values)
                    #print ("第二步完成！")
            print ("第一步完成！")

    with g2.as_default():

        output_graph_def = tf.GraphDef()

        with open(pb_file_path2, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")


        with tf.Session() as sess2:
            #init = tf.global_variables_initializer()

            #sess2.run(init)

            bottleneck_tensor2 = g2.get_tensor_by_name("BottleneckInputPlaceholder:0")
            #print ("这是第三步！",bottleneck_tensor2)
            img_out_softmax = g2.get_tensor_by_name("full_connect_layer2/Softmax:0")
            #print ("这是第四步！",img_out_softmax)

            init = tf.global_variables_initializer()

            sess2.run(init)
            length = len(bottl_values)
            print("长度是",length)
            i = 0

            for filename in os.listdir(filepath):              #listdir的参数是文件夹的路径
                print( "我看看文件夹里面的东西？",filename  )                              #此时的filename是文件夹中文件的名称
                if filename[0]!='.':
                    filename = file + filename
    
                    img_out = sess2.run(img_out_softmax,feed_dict={bottleneck_tensor2 : bottl_values[i]})
                    prediction_labels = np.argmax(img_out, axis=1)
            
                    label_lookup_path = os.path.join(dir_path+label_path)
                    proto_as_ascii_lines = tf.gfile.GFile(label_lookup_path).readlines()
                    print ("最后结果:",filename + "___" + proto_as_ascii_lines[prediction_labels[0]])
                    label_write = filename + "___" + proto_as_ascii_lines[prediction_labels[0]] + '\n'
                    ff=open(dir_path + ans_path,'r+')
                    ff.read()
                    ff.write(label_write)
                    
                    ff.close()
                    i = i+1


def recognize(sess, jpg_path ):

            init = tf.global_variables_initializer()
            sess.run(init)
    
            img = tf.gfile.FastGFile(jpg_path, 'rb').read()

            input_x = sess.graph.get_tensor_by_name("DecodeJpeg/contents:0")
            #print ("这是第一步！",input_x)
            bottleneck_tensor = sess.graph.get_tensor_by_name("pool_3/_reshape:0")
            
            img_out_bottleneck = sess.run(bottleneck_tensor,{'DecodeJpeg/contents:0': img})

            return img_out_bottleneck


            

ReadImg(file,dir_path+"tensorflow_inception_graph.pb",dir_path+"vggs.pb")


