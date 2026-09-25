# Polyp-Segmentation-CNN-Transformer-Hybrid
Automatic polyp segmentation in colonoscopy frames is critical for computer-aided diagnosis. PraNet and Polyp-PVT combine reversed-attention mechanisms and pyramid vision Transformers. Students reproduce results, analyse cross-dataset generalisation, and propose a data-efficient semi-supervised extension.

# Overleaf LaTeX Project Proposal File Link:
https://www.overleaf.com/1912964444cnxbfnkbppqg#a4a783

# Setting up the models
Copies of the PraNet and Polyp-PVT models can be found in their own, distinct directories in this repository. The datasets in use can be found inside the datasets directory. In order to run either model, make sure to run files inside the model's own directory and that the datasets directory is located at the same level as the PraNet or Polyp-PVT directory. The dependencies for the PraNet and Polyp-PVT models can be found inside envs, the exact Conda environments can also be found in the .yml files. Setting up the weights is described separately below.

# PraNet
The pretrained weights of the PraNet model should be put in `snapshots/PraNet_Res2Net/`.

Pretrained Res2Net weights should be put in `pretrained_pth/`.

The weights can be found in [the original PraNet repository](https://github.com/DengPingFan/PraNet)

# Polyp-PVT
The pretrained model of the Polyp-PVT model can be found in [the original Polyp-PVT repository](https://github.com/DengPingFan/Polyp-PVT#43-pretrained-model) and should be put in `pretrained_pth/`

The readily trained model can also be found in [the original Polyp-PVT repository](https://github.com/DengPingFan/Polyp-PVT#47-well-trained-model) and should be put in `mode_pth/`
