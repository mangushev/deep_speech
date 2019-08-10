# deep_speech_1
Implementation of Deep Speech: Scaling up end-to-end speech recognition. This is arXiv:1412.5567v2 [
cs.CL] 19 Dec 2014 article.

This is simple and strightforward implementation in a very few program files:

- prepare_libri.py preprocesses Libri datasets and outputs in tfrecords format in local file system. Generally, idea was that Google's TPU is used for training. TPU can access data located in Google storage, so tfrecord output files have to be manually copied into Google storage

- training.py is a training program implemented in tensorflow (1.14)

- utils.py contains alphabet to index and index to alpabet mapping

- constants.py defines locations of training, testing and logs. tfrecords and logs can be located locally or on Google storage. TPU requires tfrecords located in Google storage, so TPU can access them. This file is relevant to training only and it has to be adjusted. This avoid editing training.py

<b>Preparing data</b>

Files preparation:
1. Unzip Libri somewhere. I used only Libri train-clean-100, train-clean-360 and test-clean
2. prepare_libri can split tfrecords and I split train-clean-360 into four files

Processing:
1. librosa is used to read flac files directly without transforming to wav. Sample rate is 16,000, window 20ms, step 10ms, 80 bins
2. python_speech_features package is used to get filterbank energies
I used preemp 0. and kept logarithm value >= 0 for whatever reason. I do not think it is necessary, so just preempt just set to regular 0.97 and logariphm is not constrained to positive values
3. Values are normalized within sample per bin
4. Sample is padded with zeros to max_sequence_length
5. Tensor is transformed as frequency, time, channel (1)
6. tfrecord attributes are feature, feature length, label, label length - they will be needed in training.py

Sample command:
python prepare_libri.py --max_sequence_length=1729 --max_label_length=338 --partition_size=30000 --starting_position=0 --logging=INFO --files_path=data/Libri/LibriSpeech/train-clean-360 --tfrecords_file=data/train-clean-360.tfrecords

Parameters:
--starting_position - this allows to restart processing if previous processing was interrupted

--partition_size=30000 - it will break otherwise single tfrecords file into chunks, here 30000 each 

max_sequence_length value is 1729. It is selected based on lengths of libri-train-clean-100. There are few samples beyond 1729 in this dataset. Also, max_sequence_length impacts size in convolutions and RNN length. Longes samples are discarded. Probably, this value can be better adjusted based on what libri-train-clean-360 lengths are. max_label_length is just max length of label from selected samples. 

Logging=DEBUG show feature values:
2019-08-10 21:17:50,956 DEBUG logfbank shape: (1454, 80)
2019-08-10 21:17:50,962 DEBUG logfbank:
 [[-0.1111  0.6095  1.0176  1.1137  0.2373 -0.2139 ...  4.8708  4.9459  4.5214  3.8274  4.0913  4.7642]
 [-0.4393 -1.2375 -2.5525 -0.6954 -0.3398 -0.5373 ...  4.0373  5.2597  4.7472  4.5717  4.387   3.5797]
 [-2.6724 -1.0676 -0.3013  0.0096 -0.7821 -2.1233 ...  3.7777  4.0192  3.4019  4.8136  4.3992  4.3519]
 [-0.0576  0.3938 -0.0396 -0.865  -1.0934 -0.9323 ...  4.6715  4.3411  4.7329  3.8899  4.6315  4.3994]
 [-1.2696 -0.3552  0.053  -0.4427 -2.8642 -0.8151 ...  3.8045  4.7918  4.6697  4.3464  3.6351  4.6999]
 [ 0.2793  0.3827  0.2416 -0.2412 -2.1495 -2.1655 ...  4.7993  4.4882  4.9108  4.5687  4.294   4.3851]
 ...
 [ 0.7511  0.2586 -0.2046  0.4855  0.7449  0.7626 ...  5.5409  5.227   4.4371  4.0713  4.6752  4.1292]
 [ 0.5411  0.4836 -0.0675 -0.8868 -2.1085 -1.1407 ...  4.9347  5.2138  5.5542  5.3989  4.3324  4.1431]
 [ 0.9805  0.7779  1.0359  1.3197  0.7026  0.3572 ...  5.055   4.3439  4.8219  5.2912  4.4252  3.4983]
 [-0.0622 -0.1115  0.2196  0.8163  0.3678 -0.5684 ...  4.392   3.241   4.1869  4.1345  4.3802  4.0633]
 [-1.0412 -1.0565 -0.9248 -1.033  -0.6672  0.1463 ...  4.2764  4.2982  4.8398  4.8383  4.3737  4.7109]
 [-0.4292 -2.2693 -0.433   0.9699  1.1135  1.0163 ...  3.6089  3.7566  4.2182  4.4495  4.8125  4.8761]]
2019-08-10 21:17:50,969 DEBUG shape norm: (1454, 80)
2019-08-10 21:17:50,974 DEBUG norm:
 [[-1.4322 -1.3309 -1.2687 -1.2516 -1.2836 -1.3524 ... -0.6882 -0.6477 -0.8038 -1.0335 -0.9247 -0.7091]
 [-1.5016 -1.7017 -1.9657 -1.6008 -1.3961 -1.4169 ... -0.943  -0.55   -0.7316 -0.7867 -0.8253 -1.1075]
 [-1.9742 -1.6676 -1.5262 -1.4648 -1.4823 -1.7337 ... -1.0223 -0.9362 -1.1617 -0.7065 -0.8212 -0.8478]
 [-1.4209 -1.3742 -1.4751 -1.6336 -1.543  -1.4958 ... -0.7491 -0.836  -0.7362 -1.0127 -0.7431 -0.8318]
 [-1.6774 -1.5246 -1.457  -1.5521 -1.8881 -1.4724 ... -1.0141 -0.6956 -0.7564 -0.8614 -1.078  -0.7307]
 [-1.3495 -1.3764 -1.4202 -1.5132 -1.7488 -1.7421 ... -0.7101 -0.7902 -0.6793 -0.7877 -0.8566 -0.8366]
 ...
 [-1.2497 -1.4013 -1.5073 -1.3729 -1.1847 -1.1573 ... -0.4834 -0.5602 -0.8308 -0.9526 -0.7285 -0.9227]
 [-1.2941 -1.3562 -1.4805 -1.6378 -1.7408 -1.5375 ... -0.6687 -0.5643 -0.4736 -0.5124 -0.8437 -0.918 ]
 [-1.2011 -1.2971 -1.2651 -1.2119 -1.193  -1.2383 ... -0.6319 -0.8351 -0.7077 -0.5481 -0.8125 -1.1349]
 [-1.4218 -1.4757 -1.4245 -1.309  -1.2582 -1.4232 ... -0.8346 -1.1785 -0.9107 -0.9316 -0.8276 -0.9449]
 [-1.629  -1.6654 -1.6479 -1.666  -1.4599 -1.2804 ... -0.8699 -0.8493 -0.702  -0.6983 -0.8298 -0.727 ]
 [-1.4995 -1.9089 -1.5519 -1.2794 -1.1129 -1.1067 ... -1.0739 -1.0179 -0.9007 -0.8272 -0.6823 -0.6715]]
2019-08-10 21:17:51,050 INFO data/Libri/LibriSpeech/dev-clean/1462/170138/1462-170138-0000.flac 1454 173 1 0




