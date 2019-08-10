import tensorflow as tf
import tensorflow.compat.v1.logging as logging
import numpy as np
import re
import glob
import os
import argparse
import sys
import operator
from python_speech_features import logfbank, fbank
import librosa

from utils import char_to_ix

FLAGS = None

#these expect how Libri is organized
label_mask = "{}-{}.trans.txt"
reg_mask = r'([0-9-]+)\s+(.+)'
audio_mask = '{}/**/*.flac'

np.set_printoptions(edgeitems=6, linewidth=10000, precision=4, suppress=True)

def get_features(path):
    sig, fs = librosa.load(path, sr=FLAGS.sample_rate, mono=True)

    sig = sig * 32768 #must be +-32k, it is not [-1, 1]

    #preemh is 0. 
    feat,energy = fbank(sig, samplerate=fs, winlen=FLAGS.winlen, winstep=FLAGS.winstep, nfilt=FLAGS.nrof_fbanks,
            nfft=512*2, lowfreq=125, highfreq=7600, preemph=0.97, winfunc=np.hamming)

    output_floor = -100.
    log_mel = np.log(np.maximum(float(output_floor), feat))
    
    logging.debug ("logfbank shape: {}".format(log_mel.shape))
    logging.debug ("logfbank:{} {}".format("\n", log_mel))

    #those are get data over bins, not over frame data
    mu = np.mean(log_mel, axis=0, keepdims=True)
    stdev = np.std(log_mel, axis=0, keepdims=True)

    norm_log_mel = (log_mel - mu) / np.maximum(stdev, 1e-12)

    logging.debug ("shape norm: {}".format(norm_log_mel.shape))
    logging.debug ("norm:{} {}".format("\n", norm_log_mel))

    feature_len = norm_log_mel.shape[0]

    if (feature_len<FLAGS.max_sequence_length):
        for i in range(FLAGS.max_sequence_length-feature_len):
            norm_log_mel = np.concatenate((norm_log_mel,np.zeros((1, FLAGS.nrof_fbanks))), axis=0)

    norm_log_mel = norm_log_mel[:FLAGS.max_sequence_length]

    norm_log_mel = np.reshape(norm_log_mel, [FLAGS.max_sequence_length, FLAGS.nrof_fbanks, 1])
    #input for 2-D convolution is frequency, time
    return np.transpose(norm_log_mel, [1, 0, 2]), feature_len

def audio_example(features, feature_length, label, label_length):
    record = {
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(features, [-1]))),
        'feature_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_length])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        'label_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_length]))
    }

    return tf.train.Example(features=tf.train.Features(feature=record))

def read_labels(label_file):
    labels = {}
    p = re.compile(reg_mask)
    f = open(label_file)
    for line in f:
         #103-1240-0000 CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED MISSUS RACHEL LYNDE LIVED JUST WHERE THE AVONLEA MAIN ROAD DIPPED DOWN INTO A LITTLE HOLLOW FRINGED WITH ALDERS AND LADIES EARDROPS AND TRAVERSED BY A BROOK
         m = p.match(line)
         if m:
             labels[m.group(1)] = m.group(2)
    f.close()

    return (labels)

def audio_repository(files_path):
    repository = {}
    for f in glob.glob(audio_mask.format(files_path),recursive=True):
        repository[os.path.basename(f).split('.')[0]] = f

    return repository

def create_records(audio_files, label_file, tfrecords_file, file_count, record_count, writer):

    max_feature_len = -1
    max_label_len = -1
    feature_long = 0
    label_long = 0
    labels = read_labels(label_file)
    repository = audio_repository(audio_files)

    for name, label in sorted(labels.items()):
        
        file_count = file_count + 1

        if (file_count - 1 < FLAGS.starting_position):
            continue

        if (((file_count - 1) - FLAGS.starting_position) % FLAGS.batch_size == 0):
            if (writer != None):
                writer.flush()
                writer.close()
            writer = get_writer(tfrecords_file, file_count - 1)
                    
        max_label_len = max(max_label_len, len(label))

        features, feature_len = get_features(repository[name])

        logging.info ("{} {} {} {} {}".format(repository[name], feature_len, len(label), file_count, record_count))
        
        max_feature_len = max(max_feature_len, feature_len)

        if (feature_len > FLAGS.max_sequence_length):
            logging.info ('{} skipped: {}'.format(repository[name], feature_len))
            feature_long = feature_long + 1
            continue

        features = np.float32(features)

        label_tensor = np.zeros((FLAGS.max_label_length), dtype=np.int32)
        try:
            for i, ch in enumerate(label.lower()):
                label_tensor[i] = char_to_ix[ch]
        except Exception as e:
            logging.info ("label length: {} skipped (could be character issue)".format(len(label)))
            label_long = label_long + 1
            continue

        tf_example = audio_example(features, feature_len, label_tensor, len(label))
        writer.write(tf_example.SerializeToString())
        record_count = record_count + 1

    return file_count, record_count, writer, feature_long, label_long, max_feature_len, max_label_len

def get_writer(tfrecords_file, starting_position):
    new_file = "{}.{:08d}".format(tfrecords_file, starting_position)
    logging.info ("New reoords file: {} starting position {}".format(new_file, starting_position))
    return (tf.io.TFRecordWriter(new_file))

def main(_):

    tf.compat.v1.logging.set_verbosity(FLAGS.logging)

    logging.info ("Running with parameters: {}".format(FLAGS))

    file_count = 0
    record_count = 0
    feature_long_count = 0
    label_long_count = 0
    max_feature_len = -1
    max_label_len = -1
    files_path = FLAGS.files_path 
    tfrecords_file = FLAGS.tfrecords_file 

    writer = None
    for speaker in os.listdir(files_path):
        for chapter in os.listdir(os.path.join(files_path,speaker)):
            chapter_path = os.path.join(files_path,speaker,chapter)
            labels = os.path.join(chapter_path, label_mask.format(speaker, chapter))
            file_count, record_count, writer, feature_long, label_long, feature_len, label_len = create_records(chapter_path, labels, tfrecords_file, file_count, record_count, writer)
            feature_long_count = feature_long_count + feature_long
            label_long_count = label_long_count + label_long
            max_feature_len = max(max_feature_len, feature_len) 
            max_label_len = max(max_label_len, label_len)

    logging.info ("file counr {} record count {} feature long {} label long {} max feature len {} max label len {}".format(file_count, record_count, feature_long_count, label_long_count, max_feature_len, max_label_len))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--max_sequence_length', type=int, default=512,
            help='Length of the autio signal in frames. Shorter signals will be complemented with zero filled frames, longer will be cut.')
    parser.add_argument('--sample_rate', type=int, default=16000,
            help='Signal will be resampled to this rate.')
    parser.add_argument('--nrof_fbanks', type=int, default=80,
            help='This is number of mel filter banks as per Deep Speech 1 article.')
    parser.add_argument('--winlen', type=float, default=0.020,
            help='Audio frame window size as per Deep Speech 1 article.')
    parser.add_argument('--winstep', type=float, default=0.010,
            help='Audio frame sliding as per Deep Speech 1 article.')
    parser.add_argument('--max_label_length', type=int, default=80,
            help='Max length of output strings in characters will shorter strings filled with zeros.')
    parser.add_argument('--starting_position', type=int, default=0,
            help='At what valid record to start processing.')
    parser.add_argument('--batch_size', type=int, default=20000,
            help='Size of the batch of records to split.')
    parser.add_argument('--logging', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'],
            help='Enable excessive variables screen outputs.')
    parser.add_argument('--files_path', type=str, default='data/Libri/LibriSpeech/dev-clean',
            help='Location of specific unzipped Libri file collectiob.')
    parser.add_argument('--tfrecords_file', type=str, default='data/dev-clean.tfrecords',
            help='tfrecords output file. It will be used as a prefix if split.')

    FLAGS, unparsed = parser.parse_known_args()

    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)

