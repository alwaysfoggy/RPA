# RPA #

## About ##
Corresponding code to the paper "Enhancing the transferability of adversarial examples with ramdom patch".

### Prepare pretrained models ###
* [Normlly trained models]( https://github.com/tensorflow/models/tree/master/research/slim)
* [Ensemble adversarial trained models]( https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models)
* [Visual transformers models] (https://github.com/rwightman/pytorch-image-models)
### Running attacks ###
*RPA
 `python attack.py --model_name vgg_16 --attack_method RPA --layer_name vgg_16/conv3/conv3_3/Relu --ens 60 --probb 0.7`


## Acknowledgments ##
Code refers to [FIA](https://github.com/hcguoO0/FIA)