# coding=UTF-8
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile


import tensorflow.python.platform
from six.moves import urllib
import tensorflow as tf

#from tensorflow.models.image.cifar10 import cifar10_input
#import cifar10_input  # 输入数据
import dataset
import readModel

FLAGS = tf.app.flags.FLAGS
TOWER_NAME = 'tower'
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'cifar10_data/',
                           """Path to the CIFAR-10 data directory.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = dataset.IMAGE_SIZE
NUM_CLASSES = dataset.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = dataset.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = dataset.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.



def _activation_summary(x):  # 用于查看稀疏性
  """Helper to create summaries for activations. （激活值）

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)                    # 这几个summary都是图表有关的
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x)) #输出数值的分布直接反应神经元的活跃性，如果全是很小的值说明不活跃


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var



def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian 【 正态分布的标准差  】
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,  # 在cpu上创建变量
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')  # 权重衰减项 # mul 是乘法 增加L2范式稀疏化  tf.nn.l2_loss(var) = sum(var ** 2) / 2
    # wd 相当于最后正则项前的系数，如果为0则没有正则项
    tf.add_to_collection('losses', weight_decay)  # 收集到losses里面 ，后面交叉熵也一样收集到了losses里面
  return var


def distorted_inputs():  #数据输入 （训练）
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):#数据输入  （测试）
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)


def inference(images):  #用于预测（这里有网络的构架）
  """Build the model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate（实例化） all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope: #在一个作用域 scope 内共享conv1这个变量
  # 执行完 with 里边的语句之后，这个 conv1/ 和 conv2/ 空间还是在内存中的。这时候如果再次执行上面的代码
  # 就会再生成其他命名空间
    kernel = _variable_with_weight_decay('weights', shape=[5, 5, 2048, 2048],  # 维度不升高
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [2048], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)  # 这个函数是将biases加到conv上面，也就是得到了wx+b的结果z
    conv1 = tf.nn.relu(bias, name=scope.name)  # relu激活函数
    _activation_summary(conv1)
  print("---------conv1___down",conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  print("---------pool1___down",pool1)


  # norm1 局部响应归一化函数  【简言之就是只沿depth这个维度进行规范化处理】
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
  print("---------norm1___down",norm1)


  ## 第一层终于搞定啦哈哈哈哈哈  
  #  现在遗留的问题就是Kernel里面的系数不知道是怎么设置的 ： 不用定义啊，只要一开始初始化就好了
  #  之后会不断训练得到结果的  yes就是这样


  # local3   全连接层
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in norm1.get_shape()[1:].as_list():  # get_shape 相当于 get_size()
      dim *= d
    reshape = tf.reshape(norm1, [FLAGS.batch_size, dim])  # 获取展开之后的维度？？这个是什么意思？？

    weights = _variable_with_weight_decay('weights', shape=[dim, 1960], # reshape到384个节点 然后全连接到下一层
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1960], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)
  print("---------local3___down")

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[1960, 1024],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)
  print("---------local4___down")

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES],
                                          stddev=1/1024.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name) # 一样的，相加 ；tf.matmul是矩阵乘法
    _activation_summary(softmax_linear)
  print("---------softmax___down")

  return softmax_linear



def loss(logits, labels ,label_batch):  # 计算 loss function  这里是我要修改的部分
  """Add L2Loss to all the trainable variables.
  
  模型的目标函数是求交叉熵损失和所有权重衰减项的和，loss()函数的返回值就是这个值
  Add summary for for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference(). 这是分类的结果
    logits = cifar10.inference(images)
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor（张量） of type float. 这应该是一个值 或者 向量

  或者在这里就把label分成两部分
  """

  labels = tf.cast(labels, tf.int64)
  # shape = (80, )
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels, logits = logits, name = 'cross_entropy_per_example')

  print ("我看看这里面是多少！", cross_entropy)
  # 第一个参数是logits = [batchsize，num_classes]，第二个是标签
  # 只要修改这个部分就可以了
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')  # 求平均值，所有参数里面，相当于归一化的操作
  tf.add_to_collection('losses', cross_entropy_mean) #收集进losses中


  

  # The total loss is defined as the 	plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):  # 这个是用于可视化的 tensorboard 的内容
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  ** train() 函数最终会返回一个用以对一批图像执行所有计算的操作步骤，以便训练并更新模型 **

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  # 确定学习率
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, #根据当前的训练步数、衰减速度、之前的学习速率确定新的学习速率
                                  global_step,           #这个函数叫什么……指数衰减法
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR, 
                                  staircase=True)
  #大概公式是这样： decayed_learning_rate=INITIAL_LEARNING_RAT*decay_rate^(global_step/decay_steps)  
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss) # 这个应该是添加到图里面去

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]): #设计是用来控制计算流图的，给图中的某些计算指定顺序
    opt = tf.train.GradientDescentOptimizer(lr) # 也就是说，只有loss_averages_op计算了之后，opt,grads才可以计算
    #实例化了一个优化函数 opt
    grads = opt.compute_gradients(total_loss) # ??

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step) # 反向传播 更新参数

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():   # 先把数据集下载下来  main函数第一句就调用了这个
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.mkdir(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
