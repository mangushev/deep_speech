# deep_speech_1
Implementation of Deep Speech: Scaling up end-to-end speech recognition. This is arXiv:1412.5567v2 [
cs.CL] 19 Dec 2014 article.

This is simple and strightforward implementation in a very few program files:

- prepare_libri.py preprocesses Libri datasets and outputs in tfrecords format in local file system. Generally, idea was that Google's TPU is used for training. TPU can access data located in Google storage, so tfrecord output files have to be manually copied into Google storage
- training.py is a training program implemented in tensorflow (1.14)
- utils.py contains alphabet to index and index to alpabet mapping
- constants.py defines locations of training, testing and logs. tfrecords and logs can be located locally or on Google storage. TPU requires tfrecords located in Google storage, so TPU can access them. This file is relevant to training only and it has to be adjusted. This avoid editing training.py

##Preparing data

Files preparation:
1. Unzip Libri. I used only Libri train-clean-100, train-clean-360 and test-clean
2. prepare_libri can split tfrecords. 100k samples will be split into 4 files: 30k, 30k, 30k, 10k if partition size is 30k

Processing:
1. librosa is used to read flac files directly without transforming to wav. Sample rate is 16,000, window 20ms, step 10ms, 80 bins
2. python_speech_features package is used to get filterbank energies
I used preemp 0. and kept logarithm value >= 0 for whatever reason. I do not think it is necessary, so just preempt just set to regular 0.97 and logariphm is not constrained to positive values
3. Values are normalized within sample per bin
4. Sample is padded with zeros to max_sequence_length
5. Tensor is transformed as frequency, time, channel (1)
6. tfrecord attributes are feature, feature length, label, label length - they will be needed in training.py

Sample command:<br>
python prepare_libri.py --max_sequence_length=1729 --max_label_length=338 --partition_size=30000 --starting_position=0 --logging=INFO --files_path=data/Libri/LibriSpeech/train-clean-360 --tfrecords_file=data/train-clean-360.tfrecords

Parameter notes (see all parameters in prepare_libri.py):
- starting_position - this allows to restart processing if previous processing was interrupted
- partition_size=30000 - it will break otherwise single tfrecords file into chunks, here 30000 each 
- max_sequence_length value is 1729. It is selected based on lengths of libri-train-clean-100. There are few samples beyond 1729 in this dataset. Also, max_sequence_length impacts size in convolutions and RNN length. Longes samples are discarded. Probably, this value can be better adjusted based on what libri-train-clean-360 lengths are. max_label_length is just max length of label from selected samples.
- logging=DEBUG shows feature values

After train and test data is prepared, optionally copy files to Google's storage:

gsutil -m cp data/*.tfrecords.* gs://robotic-gasket-999999.appspot.com/data
- 999999 - this with be specific to your storage

##Training

Flow:

- Define tfrecorddataset and iterator
- Define model
- Define loss (ctc_loss), optimizer (Adam), training op
- Specify ctc beam search
- Train
- Test - use beam search output as input for wer





