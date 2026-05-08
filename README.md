### Install

* Create a conda virtual environment and activate it:

```bash
conda create -n vitae python=3.7 -y
conda activate vitae
```

```bash
conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.2 -c pytorch -c conda-forge
```

* Install `timm==0.4.12`:

```bash
pip install timm==0.4.12
```

* Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git reset --hard a651e2c24ecf97cbf367fd3f330df36760e1c597
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp\_ext" --global-option="--cuda\_ext" ./
```

* Install other requirements:

```bash
pip install pyyaml ipdb
```

### Data Prepare
 The file structure should look like:

```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
### Evaluation
```bash
python validate.py \[ImageNetPath] --model ViTAE\_basic\_Tiny --eval\_checkpoint \[Checkpoint Path]
```
### Training
```bash
python -m torch.distributed.launch --nproc\_per\_node=4 main.py \[ImageNetPath] --model ViTAE\_basic\_Tiny -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp
```
