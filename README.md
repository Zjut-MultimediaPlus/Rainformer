# Rainformer
The rainformer is a pytorch-based encoder-forecaster model for precipitation nowcasting.

For more information or paper, please refer to [Rainformer](https://ieeexplore.ieee.org/abstract/document/9743916).

# The short introduction of files
**train_seq.npy & test_seq.npy**: the files of defining the order of data.

**tool.py**: This file contains some preprocessing function, such as data transfer function, evaluate function, show picture function, etc.

**Rainformer/Attention.py**: This file implements the channel-attention module and spatial-attention module.

**Rainformer/Rainformer.py**: This file is the kernel file, it builds the whole model, contain the local-attention module and global-attention moudle and gate fusion module.

**Rainfromer/SwinTransformer.py**: This file implements the Swin-Transformer.

**Rainformer/test.py & train.py**: The former contains the test process of the model. The train.py contains the train process of the model.

# Train
Firstly you should apply for the KNMI dataset, you can apply for the dataset by [KNMI](https://github.com/HansBambel/SmaAt-UNet).

Then, you can use Rainformer/Rainformer/train.py to train your new model or load the pre-trained model.

# Test
You can use Rainformer/Rainformer/test.py to test your model.

# Environment
Python 3.6+, Pytorch 1.0 and Ubuntu.

# Citation
 ```
@ARTICLE{9743916,
  author={Bai, Cong and Sun, Feng and Zhang, Jinglin and Song, Yi and Chen, Shengyong},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3162882}}
  ```
