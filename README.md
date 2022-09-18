# RPA #

## About ##
Corresponding code to the paper "Enhancing the transferability of adversarial examples with ramdom patch" (IJCAI 2022).


## Quick Start ##
### Prepare pretrained models ###
* [Normlly trained models]( https://github.com/tensorflow/models/tree/master/research/slim)
* [Adversarial trained models]( https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)
* [Visual transformers models](https://github.com/rwightman/pytorch-image-models)
### Running attacks ###

* RPA

`python attack.py --model_name vgg_16 --attack_method RPA --layer_name vgg_16/conv3/conv3_3/Relu --ens 60 --probb 0.7`

### Evaluate the success rate ###
* Normlly trained models nad Adversarial trained models

`python verify_tf_model.py`

* Visual transformers models

`python verify_vits.py`

## Acknowledgments ##

Code refers to [FIA](https://github.com/hcguoO0/FIA)
## Citing this work ##

@inproceedings{zhang2022enhancing,
  title={Enhancing the transferability of adversarial examples with ramdom patch},
  author={Zhang, Yaoyuan and Tan, Yu-an and Chen, Tian and Liu, Xinrui and Zhang, Quanxin and Li, Yuanzhang},
  booktitle={IJCAI},
  year={2022}
  }
