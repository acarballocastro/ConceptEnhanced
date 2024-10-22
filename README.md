# Exploiting Interpretable Capabilities with Concept-Enhanced Diffusion and Prototype Networks

This code implements Concept-Guided Conditional Diffusion, Concept-Guided ProtoPNet, and Concept-Guided ProtoPools.

## Datasets

Used datasets were:
* Caltech CUB_200_2011, available at: https://www.vision.caltech.edu/datasets/cub_200_2011/
* Animals with Attributes 2, available at: https://cvml.ista.ac.at/AwA2/

## Structure

### Concept-Guided Conditional Diffusion

This part is adapted from previous code available here: https://github.com/tcapelle/Diffusion-Models-pytorch

* `datasets` contains the files to import the CUB and AwA2 datasets and prepare the data loaders
* `modules.py` contains the model implementations
* `utils.py` contains helper functions
* `ddpm_conditional_emb.py` contains the main training algorithm
* `sampling.py` contains code to sample new images from an already trained model

Command example to train a new model:

`python ddpm_conditional_emb.py --mask 0 75 --embedding_type 'embpos'`

Command example to sample new images: 

`python ddpm_conditional_emb.py --num_samples 15 --mask 0 75 --embedding_type 'embpos'`

### Concept-Guided ProtoPNet

This part is adapted from previous code available here: https://github.com/cfchen-duke/ProtoPNet/tree/master

* `datasets` contains the files to import the CUB and AwA2 datasets and prepare the data loaders
* `utils` contains different util files:
  * `densenet_features.py`, `resnet_features.py`, and `vgg_features.py` contain code to load pre-trained models from ImageNet
  * `preprocess.py`, `receptive_field.py` and `helpers.py` contain helper functions for the implementation of the model and the training
  * `find_nearest.py` contains the function that finds the closest patches to the prototypes to create the concept prototype dataset
  * `push.py` contains code to perform the pushing of prototypes, whereas `pushing.py` allows to push prototypes for an already trained model
  * `CUB_correlation.py` contains code to calculate the correlations between concepts in the CUB dataset
* `train_and_test.py` contains the main train and testing function used in the training loop
* `model.py` contains the CG-ProtoPNet model
* `main.py` contains the main training algorithm
* `prototype_dataset.py` allows the calculation of the concept prototype dataset from a pre-trained model

Command example to train a new model: 

`python main.py --base_architecture 'vgg16' --coefs_clst 0.8 --coefs_sep = -0.08 --coefs_l1 1e-4`

Command example to create the concept prototype dataset: 

`python prototype_dataset.py --modeldir 'path_to_model_directory' --model 'model_name'`

### Concept-Guided ProtoPools

This part is adapted from previous code available here: https://github.com/gmum/ProtoPool

* `datasets` contains the files to import the CUB and AwA2 datasets and prepare the data loaders
* `utils` contains different util files:
  * `densenet_features.py`, `resnet_features.py`, and `vgg_features.py` contain code to load pre-trained models from ImageNet
  * `utils.py` contains helper functions for the implementation of the model and the training
  * `find_nearest.py` contains the function that finds the closest patches to the prototypes to create the concept prototype dataset
  * `pushing.py` contains code to push prototypes for an already trained model
  * `shared_prototypes.py` contains code to calculate the number of shared prototypes between concepts
* `model.py` contains the CG-ProtoPools model
* `main.py` contains the main training algorithm
* `prototype_dataset.py` allows the calculation of the concept prototype dataset from a pre-trained model

Command example to train a new model: 

`python main.py --base_architecture 'vgg16' --clst_weight 0.8 --sep_weight = -0.08 --l1_weight 1e-4 --orth_p_weight 1 --orth_c_weight 1`

Command example to create the concept prototype dataset: 

`python prototype_dataset.py --modeldir 'path_to_model_directory' --model 'model_name'`
