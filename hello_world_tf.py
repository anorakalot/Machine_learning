import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
c = tf.constant('Hello, world!')

with tf.Session() as sess:

        print (sess.run(c))
