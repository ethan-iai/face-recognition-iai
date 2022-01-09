# face-recognition-iai
Implementation with deep learning methods like CosineFace, SphereFace, ArcFace for face recognition task in pattern recognition class in IAI BUAA.

## Features

- Various backbones and modules (SEResNet, SEResNet-IR, ResidualAttentionNetwork and CBAM...)
- Various metrics (Softmax, ArcFace, CosineFace, ShpereFace and combination...)
- Easy to use command to train model 
- available for customized dataset 

## Installation

Install required dependencies with following shell command:

```shell
pip install -r requirements.txt
```

##  Examples

### Prepare dataset

Prepare your dataset according to documentations for `torchvision.ImageFolder`. For example, split train set and validation(test) set under `faces95` as following:

```shell
├── faces95
   ├── train
   ├── val
```

Train

For example, to train `SeResNet100-IR` with `ArcFace` method and data under `faces95` on `gpu:0`:

```shell
python attack.py faces95 \
                --epochs 8000 \
                --backbone SERes100_IR \
                --metric ArcFace \
                --workers 4 \
                --feature-dim 512 \
                --n-classes 152 \
                --scale 8 \
                --batch-size 64 \
                --lr 1e-7 \
                --momentum 0.9 \
                --save-dir ./model/SeRes100_IR_ArcFace \
                --print-freq 10 \
                --save-freq 100 \
                --gpu 0
```

## Evaluation

For example, evaluate trained model under `SeRes100_IR_ArcFace/checkpoint-6000.pth.tar` with data under `faces95/val` on `gpu:0` :

```shell
python attack.py faces95 \
                    --resume SeRes100_IR_ArcFace/checkpoint-6000.pth.tar \
                    --evaluate \
                    --epochs 8000 \
                    --backbone SERes100_IR \
                    --metric ArcFace \
                    --workers 4 \
                    --feature-dim 512 \
                    --n-classes 152 \
                    --scale 8 \
                    --batch-size 64 \
                    --lr 1e-7 \
                    --momentum 0.9 \
                    --save-dir ./model/SeRes100_IR_ArcFace \
                    --print-freq 10 \
                    --save-freq 100 \
                    --gpu 0 
```

