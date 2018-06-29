
# Model that trains/tests from the tfrecords for Astronomy

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import main_op
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants

import tensorflow as tf

FLAGS = None

import numpy      as np
import tensorflow     as tf

from keras        import utils as np_utils
from os.path      import join as os_path_join


img_shape = 21
num_images = 5



def decode(serialized_example):
  """Parses an image and label from the given serialized_example."""
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_raw'], tf.float32)
  label = tf.cast(features['label'], tf.int32)
  # image.set_shape((21*21*5))
  image = tf.reshape(image, [5, 21, 21])
  
  # Convert label from a scalar uint8 tensor to an int32 scalar.     label =
  tf.cast(features['label'], tf.int32)     
  return image, label



def normalize(image, label):
  """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
  # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  image = tf.cast(image, tf.float32)
  image = tf.transpose(image, perm=[2,1,0])
  maxval = tf.reduce_max(image, keepdims=True, axis=[0,1])
  minval = tf.reduce_min(image, keepdims=True, axis=[0,1])
  image = (image - minval)/ (maxval-minval) 
  return image, label



def augment(image, label):
  # Converting the shape
  # image = tf.reshape(image, [21, 21, 5])
  label = tf.one_hot(label, 2)
  # tf.Print(label, [label], "label looks like thisl")
  return image, label



# TODO: Have a aggregate accuracy
def Test(model_dir, test_fl, batch_size_val=100):
  # declar this to make saver work
  v1 = tf.get_variable("v1", shape=[3])

  saver = tf.train.Saver()
  # try:
  with tf.Session(graph=tf.Graph()) as sess:
    new_saver = tf.train.import_meta_graph( os_path_join(model_dir, 'MyModel.meta'))
    
    num_epochs = 1
    # filename = os_path_join(out_dir, "Deep_22_15k.tfrecords")
    # TFRecordDataset opens a binary file and reads one record at a time.
    # `filename` could also be a list of filenames, which will be read in order.

    graph = tf.get_default_graph()
    new_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    y_conv = graph.get_tensor_by_name("y_conv:0")
    filename = graph.get_tensor_by_name("filename:0")
    batch_size = graph.get_tensor_by_name("batch_size:0")
    num_epochs = graph.get_tensor_by_name("num_epochs:0")
    # dataset = tf.data.TFRecordDataset(filename)
    # dataset = dataset.map(decode)
    # dataset = dataset.map(augment)
    # dataset = dataset.map(normalize)

    # dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(batch_size)
    
    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction, name='accuracy')
    
    dataset_init_op = graph.get_operation_by_name('dataset_init')
     
    sess.run(dataset_init_op, feed_dict={filename: test_fl.split(), batch_size: batch_size_val, num_epochs: 1})
    try:
      while(1):
        print(sess.run(accuracy))
        # y_conv_num, x_val, y_val = sess.run([ y_conv, x, y_])
        # print(y_conv_num)
        # print(x_val) 
        
    except tf.errors.OutOfRangeError:
      return 
    # print('test accuracy %g' % accuracy.eval()) 
  # except tf.errors.OutOfRangeError:
    # print("Done evaluating") 



def deepnn(x):
  def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='VALID')

  def max_pool(x):
    return tf.nn.max_pool3d(x, ksize=[1, 3, 3, 1, 1],
                          strides=[1, 3, 3, 1, 1], padding='VALID')

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, img_shape, img_shape, num_images, 1])

  # print("After Reshape Layer")
  # print (x_image.shape)
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([3, 3, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)

  # print("After First Conv Layer")
  # print (h_conv1.shape)

  with tf.name_scope('pool1'):
    h_pool1 = max_pool(h_conv1)

  # print("After First pool Layer")
  # print (h_pool1.shape)
  keep_prob = tf.placeholder_with_default(1.0, shape=())
  dropout1 = tf.nn.dropout(h_pool1, keep_prob)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 1, 32, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv3d(dropout1, W_conv2) + b_conv2)

  # print("After Second conv Layer")
  # print (h_conv2.shape)

  with tf.name_scope('pool2'):
    h_pool2 = max_pool(h_conv2)

  # print ("After Second pool Layer")
  # print (h_pool2.shape)

  dropout2 = tf.nn.dropout(h_pool2, keep_prob)

  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([1*1*1*32, 64])
    b_fc1 = bias_variable([64])

    h_pool2_flat = tf.reshape(dropout2, [-1, 1*1*1*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
  # print ("After fully connected layer")
  # print (h_fc1.shape)
  
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([64, 64])
    b_fc2 = bias_variable([64])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  print (h_fc1_drop.shape)
  
  with tf.name_scope('fc3'):
    W_fc3 = weight_variable([64, 2])
    b_fc3 = bias_variable([2])

    y_conv = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
  
  # with tf.name_scope('fc2'):
  #   W_fc2 = weight_variable([64, 2])
  #   b_fc2 = bias_variable([2])

  #   y_conv = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
  # print ("After second fully connected layer")
  # print (y_conv.shape)
  return y_conv, keep_prob



def Train(out_dir, fl_nm):
  with tf.Graph().as_default():
    # batch_size = 32
    batch_size = tf.placeholder(tf.int64, shape=(), name="batch_size")
    # x = tf.placeholder(tf.float32, [None, img_shape, img_shape, num_images, 1], name="x")
    # y_ = tf.placeholder(tf.float32, [None, 2], name="y_")

    filename = tf.placeholder(tf.string, shape=[None], name="filename")

    # x, y_ = inputs(batch_size, 1000)
    
    num_epochs = tf.placeholder(tf.int64, shape=(), name="num_epochs")
    # filename = os_path_join(out_dir, "Deep_22_15k.tfrecords")
    # TFRecordDataset opens a binary file and reads one record at a time.
    # `filename` could also be a list of filenames, which will be read in order.
    dataset = tf.data.TFRecordDataset(filename)

    # The map transformation takes a function and applies it to every element
    # of the dataset.
    
    dataset = dataset.map(decode)
    dataset = dataset.map(augment)
    dataset = dataset.map(normalize)
    
    # The shuffle transformation uses a finite-sized buffer to shuffle elements
    # in memory. The parameter is the number of elements in the buffer. For
    # completely uniform shuffling, set the parameter to be the same as the
    # number of elements in the dataset.
    # dataset = dataset.shuffle(1000 + 3 * batch_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    # handle = tf.placeholder(tf.string, shape=[], name="handle")
    # iterator = tf.data.Iterator.from_string_handle(
    #     handle, dataset.output_types, dataset.output_shapes)
    # iterator_ = dataset.make_initializable_iterator()

    # x, y_ = iterator_.get_next()
    iterator = dataset.make_initializable_iterator()
    
    x, y_ = iterator.get_next()
    dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')

    # y_conv = tf.placeholder(tf.float32, [None, 2], name="y_conv")

    y_conv, keep_prob = deepnn(x)

    x = tf.identity(x, name='x')
    y_ = tf.identity(y_, name='y_')
    y_conv = tf.identity(y_conv, name='y_conv')

    with tf.name_scope('loss'):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                              logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
      # train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
      train_step = tf.train.RMSPropOptimizer(1e-3).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

    saver = tf.train.Saver()
    # out_dir 
    model_path = os_path_join(out_dir, "MyModel")
    # Needs a list of files names
    tffile = fl_nm.split()
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      # training_handle = sess.run(iterator_.string_handle())
      sess.run(dataset_init_op, feed_dict={filename: tffile, batch_size: 32, num_epochs: 10, keep_prob: 0.7})
      i=0
      try:
        while(1):
          if i % 100 == 0:
            train_accuracy = accuracy.eval()
            print('step %d, training accuracy %g' % (i, train_accuracy))
            cross_entropy_val = cross_entropy.eval() 
            print("cross entropy = %g" % cross_entropy_val)
          i = i+1
          train_step.run()
      except tf.errors.OutOfRangeError:
        print("Training Complete")
      save_path = saver.save(sess, model_path)
      print("Model saved in file: %s" % save_path)



def main(opts):
  start = time.time()
  # Train(out_dir, os_path_join(src_dir, trainfile))
  if opts.Train:
    Train(opts.out_dir[0], opts.Train[0])
  else:
    Model_dir = opts.Test[1]
    Test_fl = opts.Test[0]
    batch_size_val = 100
    Test(Model_dir, Test_fl, batch_size_val)
  print("Total Time %f sec" % (time.time() - start))



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--Train', nargs=1, help='The location of the tfrecords')
  parser.add_argument('--Test', nargs=2, help='First argument the location of the test images and \
                        Second argument the location of the model')
  parser.add_argument('--out_dir', default='./', help='Output Location')
  opts = parser.parse_args()
  # opts.Train = ["/home/patel.3140/ASSASIN/Astro_Code/TestInput/Test_astro/Deep_22_15k/deep/TFRecord/Train.tfrecords"]
  # opts.out_dir = ["./DeleteThis/"]
  # opts.Test = ["/home/patel.3140/ASSASIN/Astro_Code/TestInput/Test_astro/Deep_22_15k/deep/TFRecord/Test.tfrecords",
  #                           "./DeleteThis/"]
  main(opts)