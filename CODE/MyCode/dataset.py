#encoding=utf-8
import os
import tensorflow as tf
from PIL import Image

IMAGE_SIZE = 300

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5

#cwd = os.getcwd()

#制作二进制数据
def create_record(cwdd,classes,filename,samples):
    writer = tf.python_io.TFRecordWriter(filename)
    for index, name in enumerate(classes):
        class_path = cwdd +"/"+ name+"/"
        for num in range (1,samples):
            img_path = class_path + str(num) + '.JPEG'
            img = Image.open(img_path)
            if img.mode == "RGB":
                r,g,b= img.split()
            else:
                r,g,b,a= img.split()
            img = Image.merge("RGB", (r, g, b))
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_raw = img.tobytes() #将图片转化为原生bytes
            example = tf.train.Example(
            features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    # "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[index_str])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


   

#data = create_record()

#读取二进制数据
def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label

"""
if __name__ == '__main__':
    save_path = "/Users/gaoyang5/Downloads/YUhuangyue/tensorflow.cifar10-master/res/"
    if 0:
        data = create_record("train.tfrecords2")
    else:
        img_t, label = read_and_decode("train.tfrecords2")
        print "tengxing",img_t,label
        # 使用shuffle_batch可以随机打乱输入 next_batch挨着往下取
        # shuffle_batch才能实现[img,label]的同步,也即特征和label的同步,不然可能输入的特征和label不匹配
        # 比如只有这样使用,才能使img和label一一对应,每次提取一个image和对应的label
        # shuffle_batch返回的值就是RandomShuffleQueue.dequeue_many()的结果
        # Shuffle_batch构建了一个RandomShuffleQueue，并不断地把单个的[img,label],送入队列中
        img_batch, label_batch = tf.train.shuffle_batch([img_t, label],
                                                    batch_size=200, capacity=100,
                                                    min_after_dequeue=60)
        #labels = tf.reshape(label_batch, 200)
        
        # 初始化所有的op
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            coord=tf.train.Coordinator()
            # 启动队列
            threads = tf.train.start_queue_runners(sess=sess)
            for i in range(30):
                val, l = sess.run([img_batch, label_batch])
                print "妖怪，哪里跑！：",val[i],l[i]
              #  img=Image.fromarray(val, "RGB")#这里Image是之前提到的
              #  img.save(save_path+str(i)+'_''Label_'+str(l)+'.png')#存下图片
                # l = to_categorical(l, 12)
                # print(val.shape, l)
                print "done！"   
           # coord.request_stop()
           # coord.join(threads)

"""
