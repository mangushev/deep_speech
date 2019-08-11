import time
import argparse
import sys
import re
import numpy as np
import logging
import tensorflow as tf
from tensorflow.compat.v1 import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from google.cloud import storage
from jiwer import wer

from utils import ix_to_char, nrof_classes
from constants import *

#np.set_printoptions(edgeitems=10, linewidth=10000, precision=0, suppress=True, formatter={'all':lambda x: '{:3d}'.format(int(x))})
np.set_printoptions(edgeitems=8, linewidth=10000, precision=12, suppress=True)

FLAGS = None

logger = logging.getLogger('tensorflow')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def get_lastcheckpoint(bucket_name, prefix, delimiter):

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

    last_step = -1
    p = re.compile(checkpoints_mask)
    for blob in blobs:
        m = p.match(blob.name)
        if m:
            last_step = max(int(m.group(1)), last_step)

    return "{}-{}".format(checkpoints, last_step)

def lower_convolutions(inputs):

    #calculate feature size input into RNN depending on nrof_fbanks provided (stride in frequency is 2 and conv is SAME)
    frequency_padding1 = (41 - 1) / 2
    frequency_size1 = int((FLAGS.nrof_fbanks + 2 * frequency_padding1 - 41) / 2) + 1   #40
    frequency_padding2 = (21 - 1) / 2
    frequency_size2 = int((frequency_size1 + 2 * frequency_padding2 - 21) / 2) + 1 #20
    frequency_padding3 = (21 - 1) / 2
    frequency_size3 = int((frequency_size2 + 2 * frequency_padding3 - 21) / 2) + 1 #10
    feature_size = frequency_size3 * 96

    #nrof_steps depend on stride in the first conv layer (convolution is SAME)
    context_window = FLAGS.context_size * 2 + 1
    time_padding = (context_window - 1) / 2
    nrof_steps = int((FLAGS.max_sequence_length + 2 * time_padding - context_window) / FLAGS.stride_size) + 1

    logger.debug("Last frequency dimension: {}".format(frequency_size3))
    logger.debug("RNN steps: {}".format(nrof_steps))
   
    with tf.name_scope('lower_convolutions') as scope:

        #batch, frequency bins, time (frames), 1
        with tf.name_scope('conv1') as scope:
            filter1 = tf.compat.v1.get_variable(initializer=tf.compat.v1.initializers.he_normal(), shape=[41, context_window, 1, 32], name='filter1')
            conv1 = tf.nn.conv2d(inputs,filter1,strides=[1, 2, FLAGS.stride_size, 1], padding="SAME", use_cudnn_on_gpu=True, data_format='NHWC', name='conv1')
            bias1 = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[32], name='bias1')
            conv_bias1 = tf.nn.bias_add(conv1, bias1, data_format='NHWC', name='conv_bias1')
            #tf.maximum with bfloat16 enabled causes double less_equal op registration error
            if (FLAGS.bfloat16):
                conv_bias_relu1 = tf.nn.relu6(conv_bias1, name='conv_bias_relu1')
            else:
                conv_bias_relu1 = tf.minimum(tf.maximum(conv_bias1, 0),20, name='conv_bias_relu1')

        with tf.name_scope('conv2') as scope:
            filter2 = tf.compat.v1.get_variable(initializer=tf.compat.v1.initializers.he_normal(), shape=[21, context_window, 32, 32], name='filter2')
            conv2 = tf.nn.conv2d(conv_bias_relu1,filter2,strides=[1, 2, 1, 1], padding="SAME", use_cudnn_on_gpu=True, data_format='NHWC', name='conv2')
            bias2 = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[32], name='bias2')
            conv_bias2 = tf.nn.bias_add(conv2, bias2, data_format='NHWC', name='conv_bias2')
            if (FLAGS.bfloat16):
                conv_bias_relu2 = tf.nn.relu6(conv_bias2, name='conv_bias_relu2')
            else:
                conv_bias_relu2 = tf.minimum(tf.maximum(conv_bias2, 0),20, name='conv_bias_relu2')
    
        with tf.name_scope('conv3') as scope:
            filter3 = tf.compat.v1.get_variable(initializer=tf.compat.v1.initializers.he_normal(), shape=[21, context_window, 32, 96], name='filter3')
            conv3 = tf.nn.conv2d(conv_bias_relu2,filter3,strides=[1, 2, 1, 1], padding="SAME", use_cudnn_on_gpu=True, data_format='NHWC', name='conv3')
            bias3 = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[96], name='bias3')
            conv_bias3 = tf.nn.bias_add(conv3, bias3, data_format='NHWC', name='conv_bias3')
            if (FLAGS.bfloat16):
                conv_bias_relu3 = tf.nn.relu6(conv_bias3, name='conv_bias_relu3')
            else:
                conv_bias_relu3 = tf.minimum(tf.maximum(conv_bias3, 0),20, name='conv_bias_relu2')
    
    return conv_bias_relu3, nrof_steps, feature_size

clipped_relu = lambda d: tf.minimum(tf.maximum(d, 0),20)

def deep_speech_1(features, training_mode):

    #IN: b, f, t, 1: batch, frequency bins, frames, 1
    if (FLAGS.bfloat16):
        with tf.compat.v1.variable_scope('lower_convs', dtype=tf.bfloat16, reuse=tf.compat.v1.AUTO_REUSE):
            conv_inputs, nrof_steps, feature_size = lower_convolutions(features)
    else:
        conv_inputs, nrof_steps, feature_size = lower_convolutions(features)
    #OUT: b, 10, 320, 96]

    with tf.name_scope('recurrent_proc') as scope:
   
        #bfloat16 is only used ibn convolutions
        conv_inputs1 = tf.cast(conv_inputs, tf.float32)
        #b, f, t, c -> t, b, f, c: make [t, b, f, c]
        step_inputs1 = tf.transpose(conv_inputs1, [2, 0, 1, 3])
        #t, b, f, c -> t, b, f 
        step_inputs = tf.reshape(step_inputs1, [nrof_steps, FLAGS.batch_size, feature_size], name='steps_inputs1')

        with tf.compat.v1.variable_scope('rnncells'):

            cell_fw = tf.keras.layers.SimpleRNNCell(FLAGS.recurrent_unit_size, activation=clipped_relu, kernel_initializer=tf.compat.v1.initializers.he_normal(), recurrent_initializer=tf.orthogonal_initializer, bias_initializer=tf.zeros_initializer, dropout=FLAGS.recurrent_dropout, name='forward_cell')
            cell_bw = tf.keras.layers.SimpleRNNCell(FLAGS.recurrent_unit_size, activation=clipped_relu, kernel_initializer=tf.compat.v1.initializers.he_normal(), recurrent_initializer=tf.orthogonal_initializer, bias_initializer=tf.zeros_initializer, dropout=FLAGS.recurrent_dropout, name='backward_cell')

            step = tf.constant(0)
            forward_output_ta = tf.TensorArray(size=nrof_steps, dtype=tf.float32)
            backward_output_ta = tf.TensorArray(size=nrof_steps, dtype=tf.float32)
            forward_initial_state = tf.zeros((FLAGS.batch_size, FLAGS.recurrent_unit_size), dtype=tf.float32, name='forward_state')
            backward_initial_state = tf.zeros((FLAGS.batch_size, FLAGS.recurrent_unit_size), dtype=tf.float32, name='backward_state')

            def cond(step, forward_output_ta, backward_output_ta, forward_state, backward_state):
                return tf.less(step, nrof_steps)

            def body(step, forward_output_ta, backward_output_ta, forward_state, backward_state):

                forward_input = tf.slice(step_inputs, [step, 0, 0], [1, -1, -1], name='forward_slice')
                forward_input_one = tf.squeeze(forward_input, axis=0, name='squeeze_forward')
                forward_output, forward_state = cell_fw(forward_input_one, forward_state, training=training_mode)

                backward_step = tf.add_n([nrof_steps, -step, -1], name="backward_step")
                backward_input = tf.slice(step_inputs, [backward_step, 0, 0], [1, -1, -1], name='backward_slice')
                backward_input_one = tf.squeeze(backward_input, axis=0, name='sqeeze_backward')
                backward_output, backward_state = cell_bw(backward_input_one, backward_state, training=training_mode)

                forward_output_ta = forward_output_ta.write(step, forward_output, name='forward_ta_w')
                backward_output_ta = backward_output_ta.write(backward_step, backward_output, name='backward_ta_w')

                return (step + 1, forward_output_ta, backward_output_ta, forward_state, backward_state)

            _, forward_output_ta_final, backward_output_ta_final, state, _ = tf.while_loop(cond, body, [step, forward_output_ta, backward_output_ta, [forward_initial_state], [backward_initial_state]], name='rnn_loop')

        forward_projections = forward_output_ta_final.stack(name='stack_forward_ta')
        backward_projections = backward_output_ta_final.stack(name='stack_backward_ta')
    
        #time, batch, features: add outputs as per article
        rnn_sum = tf.add(forward_projections, backward_projections, name='rnn_sum')
        #concat can be used instead of sum
        #rnn_sum = tf.concat([forward_projections, backward_projections], axis=2, name='rnn_sum')
        #rnn_sum.set_shape((nrof_steps, FLAGS.batch_size, FLAGS.recurrent_unit_size * 2))

    with tf.name_scope('dense_softmax') as scope:
        linear_projections = tf.keras.layers.Dense(nrof_classes, kernel_initializer=tf.random_normal_initializer(), bias_initializer=tf.zeros_initializer, name='linear_projections')(rnn_sum)

        projections = tf.nn.softmax(linear_projections, axis=2, name="projections")

    return nrof_steps, linear_projections, projections

def records_parser(record):

    keys_to_features = {
        "features": tf.io.FixedLenFeature([FLAGS.nrof_fbanks, FLAGS.max_sequence_length, 1], tf.float32),
        "feature_length": tf.io.FixedLenFeature((), tf.int64),
        "label": tf.io.FixedLenFeature([FLAGS.max_label_length], tf.int64),
        "label_length": tf.io.FixedLenFeature((), tf.int64)
    }

    parsed = tf.io.parse_single_example(record, keys_to_features)

    if FLAGS.bfloat16:
        features = tf.cast(parsed["features"], tf.bfloat16)
    else:
        features = parsed["features"]

    feature_length= tf.cast(parsed["feature_length"], tf.int32)
    label = tf.cast(parsed["label"], tf.int32)
    label_length= tf.cast(parsed["label_length"], tf.int64)

    return features, feature_length, label, label_length

def main(_):

    logger.setLevel(FLAGS.logging)

    logger.info("Running with parameters: {}".format(FLAGS))

    if (FLAGS.random_seed > 0):
        tf.compat.v1.set_random_seed(FLAGS.random_seed)
    
    global_step = tf.compat.v1.train.create_global_step()
    
    training_mode = tf.compat.v1.placeholder(tf.bool, shape=(), name='training_mode')

    nrof_training_batch = int(FLAGS.training_set_size / FLAGS.batch_size) + 1
    nrof_testing_batch = int(FLAGS.testing_set_size / FLAGS.batch_size) + 1

    logger.info("Classes(including blank): {}".format(nrof_classes))
    logger.debug("Traning batches per set: {}".format(nrof_training_batch))
    logger.debug("Testing batches per set: {}".format(nrof_testing_batch))

    files = tf.data.Dataset.list_files(train_data)
    training_dataset = (files.interleave(tf.data.TFRecordDataset, cycle_length=len(train_data), block_length=128, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)
        .repeat()
        .map(map_func=records_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(FLAGS.batch_size, drop_remainder=False)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

    testing_dataset = (tf.data.TFRecordDataset(test_data)
        .shuffle(buffer_size=FLAGS.shuffle_buffer_size, reshuffle_each_iteration=True)
        .repeat()
        .map(map_func=records_parser,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(FLAGS.batch_size, drop_remainder=False)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

    iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(training_dataset), tf.compat.v1.data.get_output_shapes(training_dataset))
    
    training_init_op = iterator.make_initializer(training_dataset)
    testing_init_op = iterator.make_initializer(testing_dataset)
    
    next_feature, next_feature_length, next_label, next_label_length = iterator.get_next(name='next_tfrecord')
    
    #features are formed as: frequency bins, frames, 1 
    if (FLAGS.bfloat16):
        nrof_steps, linear_projections, projections = deep_speech_1(next_feature, training_mode)
        linear_projections = tf.cast(linear_projections, tf.float32)
        projections = tf.cast(projections, tf.float32)
    else:
        nrof_steps, linear_projections, projections = deep_speech_1(next_feature, training_mode)

    next_feature_length.set_shape((FLAGS.batch_size))

    with tf.name_scope('ctc_sparse_labels') as scope:

        batch_index = tf.constant(0, name='batch_index', dtype=tf.int64)

        indices = tf.zeros([1, 2], tf.int64, name='indices')
        values = tf.zeros([1], tf.int32, name='values')

        def c(batch_index, indices, values):
            return tf.less(batch_index, FLAGS.batch_size)

        def b(batch_index, indices, values):

            label_length = next_label_length[batch_index] 

            ind = tf.reshape(tf.tile([batch_index], [label_length]), [label_length, 1])
            ran = tf.reshape(tf.range(label_length), [label_length, 1])

            new_index = tf.concat([ind, ran], 1)
            new_val = next_label[batch_index,:label_length]

            out_indices = tf.cond(tf.equal(batch_index, 0), lambda: new_index, lambda: tf.concat([indices, new_index], axis=0))
            out_values = tf.cond(tf.equal(batch_index, 0), lambda: new_val, lambda: tf.concat([values, new_val], axis=0))

            return (batch_index + 1, out_indices, out_values)

        _, label_indices, label_values = tf.while_loop(c, b, [batch_index, indices, values],
                shape_invariants=[batch_index.get_shape(), tf.TensorShape([None, 2]), tf.TensorShape([None])])

        sparse_labels = tf.sparse.SparseTensor(indices=label_indices, values=label_values, dense_shape=[FLAGS.batch_size, FLAGS.max_label_length])

    with tf.name_scope('ctc_loss_func') as scope:
        #blank symbol is at the end
        context_window = FLAGS.context_size * 2 + 1
        time_padding = (context_window - 1) / 2
        #sequence lengths are recalculated due to the stride of 2
        seq_lens = tf.cast(tf.floor((next_feature_length + tf.cast(2 * time_padding, tf.int32) - context_window) / FLAGS.stride_size) + 1, tf.int32)

        loss = tf.compat.v1.nn.ctc_loss(
            labels=sparse_labels,
            logits=linear_projections, #[frames, batch_size, num_labels]
            sequence_length= seq_lens, #this should be less max_sequence_len
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=True,
            ignore_longer_outputs_than_inputs=False,
            time_major=True
        )

    mean_loss = tf.reduce_mean(loss)
    if (FLAGS.summary):
        tf.compat.v1.summary.scalar('loss', mean_loss)

    lr = tf.compat.v1.placeholder(tf.float32, shape=(), name='lr')

    optimizer = tf.compat.v1.train.AdamOptimizer(lr, name="Adam")

    tvars = tf.compat.v1.trainable_variables()
    grads = tf.gradients(mean_loss, tvars, name='gradients')

    if (FLAGS.clip_gradients > 0):
        clipped_grads, _ = tf.clip_by_global_norm(grads, FLAGS.clip_gradients)
    
        gradients = clipped_grads
        grads_and_vars = zip(clipped_grads, tvars)
    else:
        gradients = grads
        grads_and_vars = zip(grads, tvars)

    training_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
         inputs=projections,
         sequence_length= seq_lens,
         beam_width=100, #default
         top_paths=1, #default
         merge_repeated=True #default
    )

    if (FLAGS.summary):
        summary_op = tf.compat.v1.summary.merge_all()

    saver = tf.compat.v1.train.Saver(max_to_keep=20)

    ds = tf.DeviceSpec(device_type="CPU")

    with tf.device(ds):
        if (FLAGS.tpu):
            session_target = TPUClusterResolver(tpu=FLAGS.tpu_name, zone=FLAGS.tpu_zone).get_master()
        else:
            session_target = ''

        with tf.compat.v1.Session(target=session_target) as sess:
    
            if (FLAGS.summary):
                train_writer = tf.compat.v1.summary.FileWriter(logs + '/train', sess.graph)
                test_writer = tf.compat.v1.summary.FileWriter(logs + '/test')
 
            if (FLAGS.tpu):
                sess.run(tpu.initialize_system())

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            if (FLAGS.restore):
                last_checkpoint = get_lastcheckpoint(bucket, logs_prefix + '/', '/')
                try:
                    saver.restore(sess, last_checkpoint)
                except Exception as e:
                    logger.error (str(e))
                    logger.error ("Restore failed. Halting.")
                    sys.exit(1)

            tvars = [tf.as_string(v.get_shape().as_list()) for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)]
            training_fetch = [training_op, tvars, gradients, label_indices, label_values, seq_lens, global_step, projections, loss, mean_loss, next_feature, next_label, next_label_length, next_feature_length]

            testing_fetch = [decoded, log_probabilities, projections, next_label, next_label_length]

            if (FLAGS.summary):
                training_fetch.append(summary_op)
                testing_fetch.append(summary_op)

            sess.run(training_init_op)
            while True:
                epoch_start = time.time()
                [global_step_] = sess.run(fetches=[global_step])
                epoch = global_step_ // nrof_training_batch
                learning_rate = FLAGS.learning_rate / (FLAGS.learning_rate_decay_factor**epoch)
                logger.info("epoch {:2d} lr {:6f} ".format(epoch+1, learning_rate))
                while True:
                    batch_start = time.time()
                    training_batch = global_step_ % nrof_training_batch

                    if ((training_batch+1) % FLAGS.trace_interval == 0 and FLAGS.summary):
                        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                        run_metadata = tf.compat.v1.RunMetadata()
                        _, tvars, gradients, label_indices, label_values, seq_lens, global_step_, projections, loss, mean_loss, features, labels, label_length, feature_length, train_summary = sess.run(fetches=training_fetch, options=run_options, run_metadata=run_metadata, feed_dict={lr:learning_rate,training_mode:True})
                        train_writer.add_run_metadata(run_metadata,'step{}'.format(global_step_), global_step=global_step_)
                        train_writer.add_summary(train_summary, global_step_)
                    else:
                        if (FLAGS.summary):
                            _, tvars_, gradients_, label_indices_, label_values_, seq_lens_, global_step_, projections_, loss_, mean_loss_, features_, labels_, label_length_, feature_length_, train_summary_ = sess.run(fetches=training_fetch, feed_dict={lr:learning_rate,training_mode:True})
                            train_writer.add_summary(train_summary_, global_step_)
                        else:
                            _, tvars_, gradients_, label_indices_, label_values_, seq_lens_, global_step_, projections_, loss_, mean_loss_, features_, labels_, label_length_, feature_length_ = sess.run(fetches=training_fetch, feed_dict={lr:learning_rate,training_mode:True})

                    if (FLAGS.logging == 'DEBUG'):
                        #logging.debug("trainable vars: {}".format(tvars))
                        logger.debug("Learning rate {} {:6f}".format(global_step_, learning_rate_))
                        logger.debug("Gradients: {} {}".format(global_step_, gradients_))
                        logger.debug("input shape: {}".format(features_.shape))
                        logger.debug("input: {} {}".format(global_step_, features_))
                        logger.debug("projections shape: {}".format(projections_.shape))
                        logger.debug("projections: {} {}".format(global_step_, projections_))
                        logger.debug("loss shape: {}".format(loss_.shape))
                        logger.debug("loss: {} {}".format(global_step_, loss_))
                        logger.debug("label lengths: {} {}".format(global_step_, label_length_))
                        logger.debug("label: {} {}".format(global_step_, labels_))
                        #logger.debug("sparse label: {} {}".format(global_step_, label_values_))
                        #logger.debug("sparse label indices: {} {}".format(global_step_, label_indices_))
                        logger.debug("sequence lengths: {} {}".format(global_step_, feature_length_))
                        logger.debug("adjusted seq lens: {} {}".format(global_step_, seq_lens_))
                        
                        logger.debug("label text: ")
                        for b in range(FLAGS.batch_size):
                            logger.debug([ix_to_char[labels[b,i]] for i in range(label_length[b])])

                    logger.info("epoch {:2d} step {:5d} batch {:4d} time {:0.2f}s loss {:0.2f}".format(epoch+1, global_step_, training_batch+1, time.time()-batch_start, np.minimum(mean_loss_, 2000000000)))
                    
                    if ((training_batch+1) % FLAGS.save_interval == 0):
                        saver.save(sess, checkpoints, global_step=global_step_)

                    if ((training_batch+1) % FLAGS.test_interval == 0):

                        sess.run(testing_init_op)

                        wers = []
                        test_start = time.time()
                        for testing_batch in range(nrof_testing_batch):

                            if (FLAGS.summary):
                                decoded_, log_probabilities_, projections_, labels_, label_length_, test_summary_ = sess.run(fetches=testing_fetch, feed_dict={lr:learning_rate,training_mode:False})
                                test_writer.add_summary(test_summary, global_step_)
                            else:
                                [decoded_, log_probabilities_, projections_, labels_, label_length_] = sess.run(fetches=testing_fetch, feed_dict={lr:learning_rate,training_mode:False})

                            p = np.argmax(projections_, axis=2)
                            speech = np.transpose(p, [1, 0])

                            if (FLAGS.logging == 'DEBUG'):
                                #for i in range(FLAGS.batch_size):
                                #    logger.debug([ix_to_char[speech[i,s]] for s in range(nrof_steps)])

                                logger.debug("decoded: {}".format(decoded_[0]))
                                logger.debug("log probabilities shape: {}".format(log_probabilities_.shape))
                                logger.debug("log probabilities: {}".format(log_probabilities_))

                            beamed_speech = tf.sparse.to_dense(
                                decoded_[0],
                                default_value=0,
                                validate_indices=True,
                                name='best_output'
                            )

                            i = 0
                            for b in range(FLAGS.batch_size):
                                beamed_sentense = []
                                while i < len(decoded_[0].values) and decoded_[0].indices[i,0] == b:
                                    beamed_sentense.append((ix_to_char[decoded_[0].values[i]]))
                                    i = i + 1
                                ground_truth = [ix_to_char[labels_[b,i]] for i in range(label_length_[b])]
                                error = wer(ground_truth, beamed_sentense, standardize=True)
                                if (FLAGS.logging == 'DEBUG'):
                                    logger.info("ground truth: {}".format(ground_truth))
                                    logger.info("beamed text: {}".format(beamed_sentense))
                                    logger.info("wer: {:0.2f}".format(error))
                                wers.append(error)

                        logger.info("epoch {:2d} step {:5d} testing time {:0.2f}s wer {:0.2f}".format(epoch+1, global_step_, time.time()-test_start, np.mean(wers)*100))

                        sess.run(training_init_op)

                    if training_batch+1 == nrof_training_batch:
                        break

                logger.info("epoch {:2d} step {:5d} lr {:6f} time {:0.2f} ".format(epoch+1, global_step_, learning_rate, time.time()-epoch_start))
                if epoch >= FLAGS.max_epoch:
                    break

            if (FLAGS.summary):
                train_writer.close()
                test_writer.close()

            if (FLAGS.tpu):
                sess.run(tpu.shutdown_system())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--training_set_size', type=int, default=948,
            help='Records in the training set.')
    parser.add_argument('--testing_set_size', type=int, default=130,
            help='Records in the testing set.')
    parser.add_argument('--context_size', type=int, default=5,
            help='Audio frame context length, recommended 5, 7, 9 as per Deep Speech 1 article.')
    parser.add_argument('--stride_size', type=int, default=2,
            help='Input can be strided reduces recurrent number of steps by this value.')
    parser.add_argument('--max_sequence_length', type=int, default=2560,
            help='Length of the autio signal in frames. It is defined in feature preparation tool.')
    parser.add_argument('--max_label_length', type=int, default=128,
            help='Max length of output strings in characters will shorter strings filled with zeros.')
    parser.add_argument('--nrof_fbanks', type=int, default=80,
            help='This is number of mel filter banks as per Deep Speech 1 article.')
    parser.add_argument('--recurrent_unit_size', type=int, default=512,
            help='Size of the rnn cell, see https://github.com/SeanNaren/deepspeech.pytorch/issues/78')
    parser.add_argument('--recurrent_dropout', type=int, default=0.10,
            help='Recommended 5-10% in the article.')
    parser.add_argument('--max_epoch', type=int, default=15,
            help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size', type=int, default=32,
            help='Batch size.')
    parser.add_argument('--shuffle_buffer_size', type=int, default=10000,
            help='Items are read from this buffer.')
    parser.add_argument('--prefetch_buffer_size', type=int, default=1,
            help='UNUSED. Prefetch number of batches.')
    parser.add_argument('--num_parallel_calls', type=int, default=4,
            help='UNUSED. Prefetch number of parallel calles should be about number of cores.')
    parser.add_argument('--tpu', default=False, action='store_true',
            help='Train on TPU.')
    parser.add_argument('--tpu_name', type=str, default='node-5',
            help='TPU instance name.')
    parser.add_argument('--tpu_zone', type=str, default='us-central1-c',
            help='TPU instance izone location.')
    parser.add_argument('--bfloat16', default=False, action='store_true',
            help='Use bfloat16 data type for calculations.')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
            help='Optimizer learning rate.')
    parser.add_argument('--learning_rate_decay_factor', type=float, default=1.2,
            help='Learning rate is divided by this number after every epoch.')
    parser.add_argument('--clip_gradients', type=float, default=-1.,
            help='Clip gradients to deal with explosive gradients.')
    parser.add_argument('--random_seed', type=int, default=-1,
            help='Random seed to initialize values in a grath. It will produce the same results only if data and grath did not change in any way.')
    parser.add_argument('--test_interval', type=int, default=100,
            help='Run on test data every interval steps.')
    parser.add_argument('--save_interval', type=int, default=100,
            help='Save checkpoint every interval steps.')
    parser.add_argument('--trace_interval', type=int, default=5,
            help='Train with full trace every interval steps.')
    parser.add_argument('--summary', default=False, action='store_true',
            help='Enable summary.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--restore', default=False, action='store_true',
            help='Restore last checkpoint.')

    FLAGS, unparsed = parser.parse_known_args()

    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
