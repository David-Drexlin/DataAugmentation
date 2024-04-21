## Run the Experiments 
1. pip install -r requirements.txt
1.5 Adjust hyper.yaml  
2. python3 train.py 
train.py -> Controls Pytorch Lighnting General Training
model.py -> Supervised Model Logic, in Pytorch Lighnting 

utils/yaml/XX.yaml configurations 
utils/utils e.g. loading model, yaml etc. 
utils/loss e.g. loss loaders
utils/augmentations augmentations used 

(3.) With a .ckpt model the model can be tested via python3 train.py --mode test

Models Available: 
- "resnet18"
- "resnet34"
- "resnet50"
- "vit_small_patch16_224"

Augmentations Available (Code inside ./utils/augmentations.py): 
- "no_augmentations"
- "normalization"
- "GeometricAugmentation"
- "ColorAugmentation"
- "ColorAugmentationsChannel"
 
 Losses Available: 
 - "binary_cross_entropy_with_logits"
 - "binary_cross_entropy"
 - "cross_entropy"

 Changed Torchmetrics.BinaryAccuarcy _safedivide to cpu!!
