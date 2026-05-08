<h1 align="left">Re-ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias</h1> 

```bash
conda create -n vitae python=3.7 -y
conda activate vitae
```

```bash
conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.2 -c pytorch -c conda-forge
```

- Install `timm==0.4.12`:

```bash
pip install timm==0.4.12
```

- Install `Apex`:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git reset --hard a651e2c24ecf97cbf367fd3f330df36760e1c597
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

- Install other requirements:

```bash
pip install pyyaml ipdb
```

### Data Prepare
We use standard ImageNet dataset, you can download it from http://image-net.org/. The file structure should look like:
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
python validate.py [ImageNetPath] --model ViTAE_basic_Tiny --eval_checkpoint [Checkpoint Path]
```

### Training

Take ViTAE_basic_Tiny as an example, to train the ViTAE model on ImageNet with 4 GPU and 128 batch size for each GPU (512 batch size in total), run

```bash
python -m torch.distributed.launch --nproc_per_node=4 main.py [ImageNetPath] --model ViTAE_basic_Tiny -b 128 --lr 1e-3 --weight-decay .03 --img-size 224 --amp
```

