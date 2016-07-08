
import cPickle
from numpy import ndarray
import os
from tensorflow.python.training.adam import AdamOptimizer

from firefly.utils.logging_utils import get_logger
import numpy as np
import tensorflow as tf
from firefly.utils.time_utils import time_it


def build_graph(network_input, input_shape, output_shape, batch_size):
    with tf.variable_scope('simple_cnn'):
        with tf.variable_scope('conv1'):
            conv1_weights = tf.get_variable('weights', [3, 3, input_shape[2], 32], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
            conv1_bias = tf.get_variable('bias', [32], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
            conv1 = tf.nn.conv2d(network_input, conv1_weights, [1, 1, 1, 1], padding='SAME')
            conv1_out = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias)), ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        with tf.variable_scope('conv2'):
            conv2_weights = tf.get_variable('weights', [3, 3, 32, 64], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
            conv2_bias = tf.get_variable('bias', [64], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
            conv2 = tf.nn.conv2d(conv1_out, conv2_weights, [1, 1, 1, 1], padding='SAME')
            conv2_out = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias)), ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        with tf.variable_scope('conv3'):
            conv3_weights = tf.get_variable('weights', [3, 3, 64, 128], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
            conv3_bias = tf.get_variable('bias', [128], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
            conv3 = tf.nn.conv2d(conv2_out, conv3_weights, [1, 1, 1, 1], padding='SAME')
            conv3_out = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(conv3, conv3_bias)), ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        with tf.variable_scope('fc4'):
            conv_out_shape = input_shape[0] * input_shape[1] * 128
            conv_out_flat = tf.reshape(conv3_out, [batch_size, conv_out_shape])
            fc4_weights = tf.get_variable('weights', [conv_out_shape, 1024], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
            fc4_bias = tf.get_variable('bias', [1024], tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
            fc_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(conv_out_flat, fc4_weights), fc4_bias))

        output_weight = tf.get_variable('out_weights', [1024, output_shape[0]], tf.float32, initializer=tf.truncated_normal_initializer(stddev=6e-2))
        output_bias = tf.get_variable('out_bias', output_shape, tf.float32, initializer=tf.truncated_normal_initializer(stddev=5e-2))
        output = tf.matmul(fc_out, output_weight) + output_bias
    
    return output

def add_loss(output, groundtruth):
    cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output, groundtruth))
    tf.add_to_collection('losses', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    tf.scalar_summary('cnn_loss', loss)
    return loss
    
def add_optimizer(loss):
    global_step = tf.Variable(0, trainable=False)
    optimizer = AdamOptimizer()
    grads_and_vars = optimizer.compute_gradients(loss)
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    return optimizer.apply_gradients(grads_and_vars, global_step)


class InputProvider:
    
    BASE_DIR = '/home/sanjeev/Downloads/datasets/cifar-10-batches-py/'            
    
    class InputIterator:
        
        def __init__(self, input_provider, batch_size, input_size):
            self.input_provider = input_provider
            self.offset = 0
            self.batch_size = batch_size
            self.input_size = input_size
        
        def __iter__(self):
            return self
        
        @time_it
        def next(self):
            if self.offset * self.batch_size < self.input_size:
                images, groundtruths = self.input_provider.get_next_batch(self.offset, batch_size)
                self.offset += 1
                return images, groundtruths
            else:
                StopIteration()
        
        
    def __init__(self):
        filenames = [ os.path.join(self.BASE_DIR, 'data_batch_{}'.format(i)) for i in xrange(1, 5)]
        self.batches = []
        self.images = []
        self.groundtruths = []
        
        for filename in filenames:
            data_dict = self.unpickle(filename)
            data = data_dict['data']
            labels = data_dict['labels']
            for i in xrange(10000):
                image = np.reshape(data[i], [ 3, 32, 32])
#                 viewer = ImageViewer(image.transpose((2, 1, 0)))
#                 viewer.show()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                image = ndarray.astype(image, np.float32)
                image = image.transpose((2, 1, 0))
                self.images.append(image)
            self.groundtruths += labels
        
    def get_next_batch(self, offset, batch_size):
        start_index = offset * batch_size
        images = self.images[start_index:start_index + batch_size]
        groundtruths = self.groundtruths[start_index:start_index + batch_size]
        return images, groundtruths
    
    def get_iter(self, batch_size):
        return InputProvider.InputIterator(self, batch_size, len(self.images))

    def unpickle(self, filename):
        fo = open(filename, 'rb')
        d = cPickle.load(fo)
        fo.close()
        return d

if __name__ == '__main__':
    input_provider = InputProvider()
    img_h = 32
    img_w = 32
    LOG_DIR = 'logs/' 
    LEARNED_WEIGHTS_FILENAME = 'checkpoints/learned_weights.ckpt'
     
    logger = get_logger()    
     
    epoch = 1000
    batch_size = 100
     
    rgbd_input_batch = tf.placeholder(tf.float32, [batch_size, img_h, img_w, 3], name='rgbd_input')
    groundtruth_batch = tf.placeholder(tf.int32, [batch_size], name='groundtruth')
     
    output = build_graph(rgbd_input_batch, [img_h, img_w, 3], [10], batch_size)
    loss = add_loss(output, groundtruth_batch)
    apply_gradient_op = add_optimizer(loss)
    session = tf.InteractiveSession()
    session.run(tf.initialize_all_variables())
     
    merged_summary = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(LOG_DIR, session.graph)
    saver = tf.train.Saver()
     
    for step in xrange(epoch):
        batch_iter = input_provider.get_iter(batch_size)
        for  i, batch in enumerate(batch_iter):
            images, groundtruths = batch
            logger.info('epoc:{}, batch:{}'.format(step, i))
            result = session.run([apply_gradient_op, loss, merged_summary], feed_dict={rgbd_input_batch:images,
                                        groundtruth_batch:groundtruths})
            loss_value = result[1]
            logger.info('epoc:{}, batch:{},loss :{}'.format(step, i, loss_value))
        summary_writer.add_summary(result[2], step)
        if step % 50 == 0:
            logger.info('Saving weights.')
            saver.save(session, LEARNED_WEIGHTS_FILENAME)
             
        logger.info('epoc:{}, loss:{}'.format(step, loss_value))
    session.close()

