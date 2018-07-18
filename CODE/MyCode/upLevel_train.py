# coding=UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from PIL import Image
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10
import upLevel
import dataset
import readModel

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('max_steps', 200,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


cwd = "C:/Oops!yhy/Learning/Graduation_Project/inception-v3/inception-v3/dataset/"
train_dir = "C:/Oops!yhy/Learning/Graduation_Project/inception-v3/inception-v3/my_train"
classes = {'beagle'}



def train():
  
  with tf.Graph().as_default():
    readModel.train_with_inception()
    global_step = tf.Variable(0, trainable=False)

    # 这里是模型的部分
    #images, labels = cifar10.distorted_inputs() 
    train_path = "C:/Oops!yhy/Learning/Graduation_Project/inception-v3/inception-v3/mytrain.tfrecords2"
 #   data = dataset.create_record(cwd,classes,train_path,2) 直接从模型里面读取出来
    img, label = dataset.read_and_decode(train_path)

    #inception_tensor = graph.get_tensor_by_name('mixed_10/join:0') # 取出这一层的值
    inception_tensor,img_tensor = readModel.train_with_inception()
    
    #inception_ans = sess.run([inception_tensor, {'DecodeJpeg/contents:0':img}])
    #inception_ans = tf.Variable(tf.random_normal([1, 8, 8, 2048]),name='inception_ans')
    #inception_ans = tf.placeholder(tf.float32,shape=(1, 8, 8, 2048),name='inception_ans')
    inception = tf.reshape(inception_tensor,[8,8,2048])

    print("step（之前）!___训练成一个batch!",inception.shape) # batch_size



#开始训练batch
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(dataset.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch([inception, label],  # 这里创建了一个队列
      batch_size = FLAGS.batch_size,
      num_threads = num_preprocess_threads,
      capacity = min_queue_examples + 3 * FLAGS.batch_size,
      min_after_dequeue = min_queue_examples)
  


    labels = tf.reshape(label_batch, [FLAGS.batch_size])
    tf.summary.image('images', images)
    print("step1!___",images.shape,labels.shape)


    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = upLevel.inference(images)
    print("step3!___",logits) # batch_size

    
    # Calculate loss.
    loss = upLevel.loss(logits, labels, label_batch)
    print("step4!___",loss)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = upLevel.train(loss, global_step)
    print("step5!___")


    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())
    print("step6!___")

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    #sess = tf.Session(config=tf.ConfigProto(
    #    log_device_placement=FLAGS.log_device_placement))

    #with tf.Session() as sess:
    sess = tf.Session()
    sess.run(init)
    #inception_ans = sess.run([inception_tensor, {'DecodeJpeg/contents:0':img}])
    tf.train.start_queue_runners(sess=sess)

    #summary_writer = tf.summary.FileWriter(train_dir,sess.graph)
      #summary_writer.close()
    #val = sess.run(loss)
    #print ("傻逼",val)


    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      #feed_dict = {img_tensor:img}
      val = sess.run([inception_tensor, {'DecodeJpeg/contents:0':images}]);
      print ("看看这样行不行？",val)
      _, loss_value = sess.run([train_op, loss])
      print ("傻逼",loss_value)
      
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

      


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  #if gfile.Exists(FLAGS.train_dir):
  #  gfile.DeleteRecursively(FLAGS.train_dir)
 # gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()








