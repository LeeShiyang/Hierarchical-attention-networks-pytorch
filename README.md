# [PYTORCH] Hierarchical Attention Networks for Document Classification

## Introduction

Here is my pytorch implementation of the model described in the paper **Hierarchical Attention Networks for Document Classification** [paper](https://www.cs.cmu.edu/%7Ediyiy/docs/naacl16.pdf). 

## Datasets:

Statistics of datasets I used for experiments. These datasets could be download from [link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)

| Dataset                | Classes | Train samples | Test samples |
|------------------------|:---------:|:---------------:|:--------------:|
| AG’s News              |    4    |    120 000    |     7 600    |
| Sogou News             |    5    |    450 000    |    60 000    |
| DBPedia                |    14   |    560 000    |    70 000    |
| Yelp Review Polarity   |    2    |    560 000    |    38 000    |
| Yelp Review Full       |    5    |    650 000    |    50 000    |
| Yahoo! Answers         |    10   |   1 400 000   |    60 000    |
| Amazon Review Full     |    5    |   3 000 000   |    650 000   |
| Amazon Review Polarity |    2    |   3 600 000   |    400 000   |

## Setting:

I almost keep default setting as described in the paper. For optimizer and learning rate, there are 2 settings I use:

- **SGD** optimizer with initial learning rate of 0.01. The learning rate is halved every 3 epochs.
- **Adam** optimizer with initial learning rate of 0.001.

Additionally, in the original model, one epoch is seen as a loop over batch_size x num_batch records (128x5000 or 128x10000 or 128x30000), so it means that there are records used more than once for 1 epoch. In my model, 1 epoch is a complete loop over the whole dataset, where each record is used exactly once.

## Training

If you want to train a model with common dataset and default parameters, you could run:
- **python train.py -d dataset_name**: For example, python train.py -d dbpedia

If you want to train a model with common dataset and your preference parameters, like optimizer and learning rate, you could run:
- **python train.py -d dataset_name -p optimizer_name -l learning_rate**: For example, python train.py -d dbpedia -p sgd -l 0.01

If you want to train a model with your own dataset, you need to specify the path to input and output folders:
- **python train.py -i path/to/input/folder -o path/to/output/folder**

## Test

For testing a trained model with your test file, please run the following command:
- **python inference.py -i path/to/test/file -p path/to/trained/model -o path/to/output/file**

You could find some trained models I have trained in [link](https://drive.google.com/open?id=1zzC4r0nn8yInWjCbVrVZPFYyOWJQizqh)

## Experiments:

I run experiments in 2 machines, one with NVIDIA TITAN X 12gb GPU and the other with NVIDIA quadro 6000 24gb GPU. For small and large models, you need about 1.6 gb GPU and 3.5 gb GPU respectively.

Results for test set are presented as follows:  A(B)/C/D:
- **A** is accuracy reproduced here.
- **B** is accuracy reported in the paper.
- **C** is the optimizer used. **S** is SGD with initial learning of 0.01, while **A** is Adam with initial learning rate of 0.001.
- **D** is the epoch when maximum accuracy observed.

Each experiment is run over 10 epochs.

|      Size     |       Small    |     Large    |
|:---------------:|:------------------:|:------------------:|
|    ag_news    | 88.20(84.35)/A/4 | 88.17(87.18)/A/2 |
|   sogu_news   | 94.95(91.35)/A/9 | 95.48(95.12)/A/6 |
|    db_pedia   | 97.58(98.02)/A/8 | 97.65(98.27)/A/7 |
| yelp_polarity |                  | 94.21(94.11)/A/5 |
|  yelp_review  |                  | 60.55(60.38)/S/5 |
|  yahoo_answer | 67.55(70.16)/S/9 | 68.87(70.45)/S/4 |
| amazon_review |                  |                  |
|amazon_polarity|                  |                  |

You could find detail log of each experiment containing loss, accuracy and confusion matrix at the end of each epoch in **output/datasetname_scale/logs.txt**, for example output/ag_news_small/logs.txt
