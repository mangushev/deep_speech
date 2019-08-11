#TPU will require google storage, create you own bucket
bucket = 'robotic-gasket-999999.appspot.com'
#if TPU is not used, train_data can be local folder
train_data = [
    'gs://' + bucket + '/data/train-clean-360-f1729-l338.tfrecords.00000000',
    'gs://' + bucket + '/data/train-clean-360-f1729-l338.tfrecords.00030000',
    'gs://' + bucket + '/data/train-clean-360-f1729-l338.tfrecords.00060000',
    'gs://' + bucket + '/data/train-clean-360-f1729-l338.tfrecords.00090000',
    'gs://' + bucket + '/data/train-clean-100-f1729-l338.tfrecords'
    ]
#if TPU is not used, test_data can be local folder
test_data = 'gs://' + bucket + '/data/test-clean-f1729-l338.tfrecords'
logs_prefix = 'logs'
#if TPU is not used, logs can be local folder
logs_location = 'gs://' + bucket + '/' + logs_prefix
checkpoints = logs_location + '/deep_speech_1.ckpt'
checkpoints_mask = logs_prefix + r'\/deep\_speech\_1\.ckpt\-([0-9]+)\.index'
