# Deep Speech - implementation of 1412.5567v2 19 Dec 2014 article
Article link: https://arxiv.org/abs/1412.5567

This is a simple and strightforward implementation in a very few program files. In general, code does not use depricated tensorflow elements: 
- prepare_libri.py preprocesses Libri datasets and outputs in tfrecords format in local file system. Generally, the idea was that Google's TPU will used for training. TPU can access data located in Google storage, so tfrecord output files have to be copied into Google storage
- training.py is a training program implemented in tensorflow (1.14)
- utils.py contains alphabet to index and index to alpabet mapping
- constants.py defines file locations for training, testing and logs. tfrecords and logs can be located locally or on Google storage. TPU requires tfrecords located in Google storage, so TPU can access them. This file is relevant to training.py only and it has to be adjusted. This avoids editing training.py
## Required packages and data
- librosa: reading flac files. librosa may require in turn some other packages, just follow what it asks. Install: pip install librosa
- python_speech_features: filterbank energies. Install: pip install python_speech_features
- jiwer to calculate wer. Install: pip install jiwer
- Libri: http://www.openslr.org/12/
## Preparing data
Files preparation:
1. Unzip Libri. I used Libri train-clean-100, train-clean-360 and test-clean only
2. prepare_libri.py can split tfrecord ouput files. For example, 100k of samples can be split into 4 files: 30k, 30k, 30k, 10k if partition size parameter is 30k. Output generation can restart with starting_position parameter. This is very helpful if Google preemtive instance is used

Processing:
1. librosa is used to read flac files directly without transforming to wav. Sample rate is 16,000, window 20ms, step 10ms, 80 bins
2. python_speech_features package is used to get filterbank energies
I used preemp 0. and kept logarithm value >= 0. I do not think it is necessary, so preemph is set to regular 0.97 and logariphm is not constrained to positive values. You may experiment with parameters and see resulting feature values using logging=DEBUG
3. Values are normalized within sample per bin
4. Sample is padded with zeros to max_sequence_length
5. Tensor is transformed as frequency (bins), time (frames), channel (1)
6. tfrecord attributes are feature, feature length, label, label length - they will be needed in training.py

Sample command:\
`python prepare_libri.py --max_sequence_length=1729 --max_label_length=338 --partition_size=30000 --starting_position=0 --logging=INFO --files_path=data/Libri/LibriSpeech/train-clean-360 --tfrecords_file=data/train-clean-360.tfrecords`

Parameter notes (see all parameters in prepare_libri.py):
- starting_position - this allows to restart processing if previous processing was interrupted
- partition_size=30000 - it will break otherwise single tfrecords file into chunks, here 30000 samples each 
- max_sequence_length value is 1729. It is selected based on lengths of libri-train-clean-100. There are few samples beyond 1729 in this dataset. Also, max_sequence_length impacts convolutions weight sizes and RNN length. Longer samples are discarded. Probably, this value can be better adjusted based on what libri-train-clean-360 lengths are. max_label_length is just max length of label from selected samples.
- logging=DEBUG shows feature values

Length distribution in traning-clean-360 dataset:\
![traning-clean-360](images/training_set_lengths.png "traning-clean-360")

After train and test data is prepared, optionally copy files to Google's storage:\
`gsutil -m cp data/\*.tfrecords.\* gs://robotic-gasket-999999.appspot.com/data`
- 999999 - this with be specific to your storage

Hardware:
- Compute instance can be just 2 vCPU and 7.5GB memory. Since prepare_libri.py is restartable, just leave it running until done without fear to be preempted

## Training
Program flow:

1. Defining tfrecorddataset and iterator
2. Defining model
3. Defining ctc_loss, optimizer (Adam) and training op
4. Providing ctc beam search
5. Looping over traning and testing data

### 
1. Model is fed from tfrecord files. It is simple and efficient mechanism without doing much custom development. tf.data.dataset provides inteleave, shuffle, prefetch, map options removing burden doing this manually. Also, it is more efficient compare to feed dictionary. All prepare_libri tfrecord files are fed during traning. Traning and testing dataset are switched as per testing_interval parameter
2. Model is defined as per article:
- 3 convolutional layers: stride 2 in time dimention of first layer, context size is 5 so filter will be 5\*2+1=11, input is batch, frequency (bins), time (frames), channel, clipped relu (issue with bfloat16 discussed later)
- bidirectional RNN: implemented with while_loop and simplernncell, dropout should be 5-10%, recurrent unit size 512, both outputs are summed, but could be concatenated as it seems can better preserve information
- dense layer: size is number of classes including blank (29 total size)
- model returns both linear and softmax. Linear output is needed for ctc_loss function
3. ctc_loss:
- sequence lengths are recalculated due to stride 2 in convolutuions
- labels provided as sparse tensor and implemented as while_loop (there is somewhere implementation available already)
- logits are linear data in a form of time, batch, probabilities
- Adam optimizer, learning rate I used constant 0.0003, but this is configurable to reduce by factor on every epoch
- gradient clipping works on float32
4. beam serch provides the best estimate. Beam search output is used for wer calculation
5. Training and testing
- Article says to loop over 10-15 epochs. To calculate epochs as well as testing batches, accurate training_set_size and testing_set_size parameters are needed
- training runs with CPU or TPU
- mixed bfloat16 calculation can be used. bfloat16 is enabled on convolutions only, RNN did produce any speed improvement
- to run training and testing, dataset is switched
- testing invokes beam search, jiwer wer package is used to calculate error rate

Mixed presision training:
bfloat16 calculates faster in some cases. 

- It can be enabled on convolutions only, RNN did not produce any speed improvement
- bfloat16 in not fully fixed in tensorfliow 1.14. tf.maximum uses less_equal op and is has a bug with double registration which will be fixed in post 1.14. So I put just relu6 to make it work
- I was not able to overcome nan loss if I use bfloat16. This need to be looked into. Could be because of relu6

Command line sample (see all parameters in training.py):\
`python training.py --max_sequence_length=1729 --max_label_length=338 --training_set_size=132492 --testing_set_size=2472 --trace_interval=10 --save_interval=20 --test_interval=200 --shuffle_buffer_size=150000 -tpu --tpu_name='node-8' --learning_rate=0.0003 --learning_rate_decay_factor=1.`

- max_sequence_length, max_label_length should be same as in prepare_linri.py
- training_set_size, testing_set_size should be accurate for epochs calculation
- save_interval - to prevent your computing time if traning is interrupted
- test_interval - longer itervals save computing time, progress can be judged by loss
- learning rate formula: learning_rate / (learning_rate_decay_factor\*\*epoch), use factor 1 to keep lr the same

Debug:
- use logging=DEBUG to enable extensive output
- I used debug to see features, projections, gradients. Also, anything else can be added such as to see convolutions weigths or outputs

Loss, wer and test output samples:

- ctc loss - I removed some part of the image. It was due to wrong training parameters
![training loss](images/ctc-loss22.png "training loss")
- wer - this is how far I got. In the article they used more data for training. train-other-500 can be added into traning set 
![error rate](images/wer2.png "error rate")
- output sample:
![output sample](images/sample-output.png "output sample")

Hardware:
- I trained this on TPU: v2-8 and used both computing and TPU instance preemptable. Make sure to put script to shutdown TPU as soon as compute instance is preempted or does down:\
`#!/bin/bash`\
`MY_USER="your_linux_instance_login_account"`\
`echo "Shutting down!  Shutting all TPU nodes."`\
`su "${MY_USER}" -c "gcloud compute tpus stop node-8 --zone=us-central1-c --async"`\
`echo "Done uploading, shutting down."`
- If using TPU, compute instance can be just 2 vCPU and 7.5gbit memory
- I did not try GPU

## Further work
- make bfloat16 work and fix relu20
- try to speed up RNN
- add more training data such as train-other-500 set
- review beam search, there should be proper language model used to transform speech recognition character outputs into the best possible english sentense 

## References
Deep Speech: Scaling up end-to-end speech recognition
https://arxiv.org/pdf/1412.5567v2
