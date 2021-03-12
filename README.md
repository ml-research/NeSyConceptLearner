# Neuro Symbolic Concpet Learner based on Slot Attention and Set Transformer

This is the official repository for the Neuro-Symbolic Concept Learner introduced in 
[Right for the Right Concept: Revising Neuro-Symbolic Concepts by Interacting 
with their Explanations](https://arxiv.org/pdf/2011.12854.pdf) by Wolfgang Stammer, Patrick Schramowski, 
Kristian Kersting, to be published at CVPR 2021.

![Concept Learner with NeSy XIL](./figures/concept_learner.png)

This repository contains the model source code for the Neuro-Symbolic Concept Learner together with a script for training the 
Concept Learner on the [CLEVR-Hans3](https://github.com/ml-research/CLEVR-Hans) data set as a minimal example of how to 
use the model. As in the original paper the concept embedding module (Set Prediction Network with Slot Attention) was 
pretrained on the original CLEVR data. 

Files for pre-training yourself can be found in ```src/pretrain-slot-attention/``` 
(follow the instructions in the corresponding README).

Please visit the [NeSy XIL](https://github.com/ml-research/NeSyXIL) repository for the Neuro-Symbolic Explanatory 
Interactive Learning approach based on this Concept Learner to see further examples from the original paper.

## How to Run with docker on GPU:

### Dataset

First download the CLEVR-Hans3 data set. Please visit the [CLEVR-Hans](https://github.com/ml-research/CLEVR-Hans) 
repository for instrucitons on this.

### Docker

To run the eaxmple train script with the CLEVR-Hans3 data follow:

1. ```cd src/docker/```

2. ```docker build -t nesy-concept-learner -f Dockerfile .```

3. ```docker run -it -v /pathto/NeSyConceptLearner:/workspace/repositories/NeSyConceptLearner -v /pathto/CLEVR-Hans3:/workspace/datasets/CLEVR-Hans3 --name nesy-concept-learner --entrypoint='/bin/bash' --runtime nvidia nesy-concept-learner```

4. ```cd repositories/NeSyConceptLearner/src/```

5. ```./scripts/clevr-hans-concept-learner_CLEVR_Hans3.sh 0 0 /workspace/datasets/CLEVR-Hans3/``` for running on gpu 0 
with run number 0 (for saving)

## Citation
If you find this code useful in your research, please consider citing:

> @article{stammer2020right,
  title={Right for the Right Concept: Revising Neuro-Symbolic Concepts by Interacting with their Explanations},
  author={Stammer, Wolfgang and Schramowski, Patrick and Kersting, Kristian},
  journal={arXiv preprint arXiv:2011.12854},
  year={2020}
}
