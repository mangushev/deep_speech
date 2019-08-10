# deep_speech_1
Implementation of Deep Speech: Scaling up end-to-end speech recognition. This is 2014 work. There is another 2015 Deep speech 2.

This is simple strightforward iplementation basically in two files:

- prepare_libri.py preprocess Libri datasets and outputs in tfrecords format in local file system. Expectation is that Google's TPU will be used for training. TPU can access data located in Google storage, so tfrecord output files have to be manually copied somewhere in Google storage.

- models.py is a training program. I use tensorflow 1.14. 

