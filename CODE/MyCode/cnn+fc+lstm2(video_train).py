# coding=utf-8
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import xlwt
import pandas as pd


# 数据参数
MODEL_DIR = 'C:\Oops!yhy\Learning\Graduation_Project\inception-v3\inception-v3\model'  # inception-v3模型的文件夹
MODEL_FILE = 'tensorflow_inception_graph.pb'  # inception-v3模型文件名
CACHE_DIR = 'C:\Oops!yhy\Learning\Graduation_Project\inception-v3\inception-v3\data/tmp/bottleneck'  # 图像的特征向量保存地址
INPUT_DATA = 'C:\Oops!yhy\Learning\Graduation_Project\inception-v3\inception-v3\data/web/web'  # 图片数据文件夹
VALIDATION_PERCENTAGE = 10  # 验证数据的百分比
TEST_PERCENTAGE = 10  # 测试数据的百分比

# inception-v3模型参数
BOTTLENECK_TENSOR_SIZE = 5  # inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 神经网络的训练参数
LEARNING_RATE = 0.01
STEPS = 400
BATCH = 2

#定义新模型地址
pb_file_path = "vggs.pb"
#类的标签名
label_file_path = "/labels.txt"

mid_layer = 3
pre_list = ['1.jpg']

#——————————————————导入数据——————————————————————
f=open('test.csv')
df=pd.read_csv(f)     #读入股票数据
frames_train = df.iloc[:,0:8].values  #取第3-10列
frames_train = frames_train[0:7]



# 从数据文件夹中读取所有的图片列表并按训练、验证、测试分开
def create_image_lists(validation_percentage, test_percentage):
    result = {}  # 保存所有图像。key为类别名称。value也是字典，存储了所有的图片名称
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 获取所有子目录
    is_root_dir = True  # 第一个目录为当前目录，需要忽略

    f=open(MODEL_DIR + label_file_path,'w')
    index = -1;

    # 分别对每个子目录进行操作
    for sub_dir in sub_dirs:
        index = index + 1;
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下的所有有效图片
        extensions = {'jpg', 'jpeg', 'JPG', 'JPEG'}
        file_list = []  # 存储所有图像
        dir_name = os.path.basename(sub_dir)  # 获取路径的最后一个目录名字
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 将当前类别的图片随机分为训练数据集、测试数据集、验证数据集
        label_name = dir_name.lower()  # 通过目录名获取类别的名称
        print( "我来看看标签名",label_name)
        label_write = str(index)+"."+label_name+'\n'
        f.write(label_write)


        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)  # 获取该图片的名称
            chance = np.random.randint(100)  # 随机产生100个数代表百分比
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (validation_percentage + test_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据集放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }

    # 返回整理好的所有数据

    f.close()
    return result


# 通过类别名称、所属数据集、图片编号获取一张图片的地址
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]  # 获取给定类别中的所有图片
    category_list = label_lists[category]  # 根据所属数据集的名称获取该集合中的全部图片
    global pre_list
    if(category_list != []):
        pre_list = category_list
    else:
        category_list = pre_list
   
    mod_index = index % len(category_list)  # 规范图片的索引
    
    base_name = category_list[mod_index]  # 获取图片的文件名
    sub_dir = label_lists['dir']  # 获取当前类别的目录名
    full_path = os.path.join(image_dir, sub_dir, base_name)  # 图片的绝对路径

    return full_path


# 通过类别名称、所属数据集、图片编号获取特征向量值的地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index,
                          category) + '.txt'


# 使用inception-v3处理图片获取特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {
        image_data_tensor: image_data
    })
    bottleneck_values = np.squeeze(bottleneck_values)  # 将四维数组压缩成一维数组
    return bottleneck_values


# 获取一张图片经过inception-v3模型处理后的特征向量
def get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                             jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          category)

    # 如果该特征向量文件不存在，则通过inception-v3模型计算并保存
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index,
                                    category)  # 获取图片原始路径
        image_data = gfile.FastGFile(image_path, 'rb').read()  # 获取图片内容
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor,
            bottleneck_tensor)  # 通过inception-v3计算特征向量

        # 将特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 否则直接从文件中获取图片的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    # 返回得到的特征向量
    return bottleneck_values


# 随机获取一个batch图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many,
                                  category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):  #how_many就是batch个数 
        # 随机一个类别和图片编号加入当前的训练数据
        """
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65535)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category,
            jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        """
        data_index = random.randrange(6)
        bottleneck = frames_train[data_index,0:5]
        ground_truth = frames_train[data_index,5:5+3]
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    print ('-------I need to know!!',len(bottlenecks[0]),len(ground_truths[0]))
    return bottlenecks, ground_truths


# 获取全部的测试数据
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor,
                         bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    print(label_name_list)
    # 枚举所有的类别和每个类别中的测试图片
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(
                image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index, category,
                jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    # 读取所有的图片
    image_lists = create_image_lists(VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    n_classes = len(image_lists.keys())
    
    # 读取训练好的inception-v3模型
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入 1024
    bottleneck_input = tf.placeholder(
        tf.float32, [None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

    # 定义新的标准答案输入 
    ground_truth_input = tf.placeholder(
        tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义两层全连接层进行输出
    with tf.name_scope('full_connect_layer1') as scope:
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, mid_layer], stddev=0.1))
        biases = tf.Variable(tf.zeros([mid_layer]))
        logits_local = tf.nn.relu(tf.matmul(bottleneck_input, weights) + biases)  # 全连接层的输出
       # final_tensor = tf.nn.softmax(logits_local)  # 直接就softmax输出le

    with tf.variable_scope('full_connect_layer2') as scope:
        weights2 = tf.Variable(tf.truncated_normal([mid_layer, n_classes], stddev=0.1))
        biases2 = tf.Variable(tf.zeros([n_classes]))
        logits_local2 = tf.nn.relu(tf.matmul(logits_local, weights2) + biases2)  # 全连接层的输出
        final_tensor = tf.nn.softmax(logits_local2)
     

    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_local2, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        cross_entropy_mean)



    # 计算正确率(之后用来运行的)
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(
            tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))

        evaluation_step = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

    prediction_labels = tf.argmax(final_tensor, axis=1, name="output")
    #read_labels = tf.argmax(ground_truth_input, 1)
    #correct_prediction = tf.equal(prediction_labels, read_labels)
    #evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    # 训练过程
    with tf.Session() as sess:

        logpath='C:\Oops!yhy\Learning\Graduation_Project\inception-v3\inception-v3\graph'
        
        summary_writer = tf.summary.FileWriter(logpath ,sess.graph)
        init = tf.global_variables_initializer().run()

        
        
        for i in range(STEPS):
            # 每次获取一个batch的训练数据


            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training',
                jpeg_data_tensor, bottleneck_tensor)
            sess.run(
                train_step,
                feed_dict={
                    bottleneck_input: train_bottlenecks,
                    ground_truth_input: train_ground_truth
                })

            # 在验证集上测试正确率
            if i % 100 == 0 or i + 1 == STEPS:

                #summary_str = sess.run(merged)
               # summary_writer.add_summary(summary_str, i)

                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation',
                    jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(
                    evaluation_step,
                    feed_dict={
                        bottleneck_input: validation_bottlenecks,
                        ground_truth_input: validation_ground_truth
                    })
                print(
                    'Step %d : Validation accuracy on random sampled %d examples = %.1f%%'
                    % (i, BATCH, validation_accuracy * 100))

        
        # 最后在测试集上测试正确率
        print("测试数据集",image_lists)
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(
            evaluation_step,
            feed_dict={
                bottleneck_input: test_bottlenecks,
                ground_truth_input: test_ground_truth
            })
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))



        # 下面是输出测试的结果
        logits_ans = sess.run(logits_local,feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        #print ("测试数据的特征向量（中间层）",logits_ans[0][1])
        #print ("测试数据标签",test_ground_truth )

        
        
        book = xlwt.Workbook(encoding='utf-8', style_compression=0) #创建一个excel对象
        sheet = book.add_sheet('test', cell_overwrite_ok=True)
        style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on',num_format_str='#,##0.00')


        # 其中的test是这张表的名字,cell_overwrite_ok，表示是否可以覆盖单元格，其实是Worksheet实例化的一个参数，默认值是False
        # 向表test中添加数据
        index = 0
        for i in range(len(test_ground_truth)):
            for pos1 in range(mid_layer):
                sheet.write(i, pos1, float(logits_ans[i][pos1]),style0)  # 其中的'0-行, 0-列'指定表中的单元，'EnglishName'是向该单元写入的内容
            for pos2 in range(n_classes):
                sheet.write(i, pos1+pos2, float(test_ground_truth[i][pos2-1]))
            
 
        # 最后，将以上操作保存到指定的Excel文件中
        book.save('test1.xls') 
   

        #把结果保存成一个新的模型
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
        with tf.gfile.FastGFile(os.path.join(MODEL_DIR, pb_file_path), mode='wb') as f:
                f.write(constant_graph.SerializeToString())


if __name__ == '__main__':

    tf.app.run()