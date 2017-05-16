# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
#TESTTESTTESTTESTSETETESETET
#충돌충돌충돌
"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import tensorflow as tf

import cifar10


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', os.getcwd() + '/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', os.getcwd() + '/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 0,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")




def eval_once(saver, summary_writer, top_k_op, summary_op,logits,labels,images,org_image):
  """Run Eval once.

                 for i in xrange(1, 6)]
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  crack_num=0
  non_crack_num=0
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = FLAGS.num_examples#int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < org_image.get_shape().as_list()[0] and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)

        classification=sess.run(tf.argmax(logits[0],0))
        classname=["crack ","Non-crack"]
        if(classification == 0):
          crack_num+=1
        else:
          non_crack_num+=1
        
        print(classname[classification],crack_num,non_crack_num,sess.run(logits[0]))
        evaled_img = org_image[step].eval()
        #img = Image.fromarray(evaled_img)
        #img.show()
        #plt.imshow(org_image[step].eval())
        plt.imshow((255*evaled_img).astype(np.uint8))
        plt.show()
        step += 1

      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    #eval_data = FLAGS.eval_data == 'test'
    #images, labels,org_image = cifar10.inputs(eval_data=eval_data)
   # Build a Graph that computes the logits predictions from the
    # inference model.
    images, org_image= cifar10.multiply_sliced_input(img=os.getcwd() + '/fb.jpg' ,
                                                    outdir=os.getcwd()+'/sliced',
                                                    imageWidth=192,
                                                    imageHeight=192,
                                                    sliceWidth=192, 
                                                    sliceHeight=)
    print('----------------Slice & Batch finished----------------\n',
          org_image.get_shape().as_list()[0],' images\n',images)
    logits = cifar10.inference_eval(images)

    top_k_op = tf.nn.in_top_k(logits, [1], 1)
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op,logits,[0],images,org_image)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
