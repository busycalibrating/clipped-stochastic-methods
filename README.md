# Clipped Stochastic Methods for Variational Inequalities with Heavy-Tailed Noise

Code for the paper "Clipped Stochastic Methods for Variational Inequalities with Heavy-Tailed Noise" by Eduard Gorbunov, Marina Danilova, David Dobre, Pavel Dvurechensky, Alexander Gasnikov, Gauthier Gidel.
The paper is accepted to NeurIPS 2022. 
The code comes jointly with the paper.

ArXiv: https://arxiv.org/abs/2206.01095

We leverage existing codebases which implement WGAN-GP and StyleGAN2, specifically:

- https://github.com/w86763777/pytorch-gan-collections
- https://github.com/NVlabs/stylegan3

As a result, instead of tarballing these full repos with our modifications applied directly (which
makes it difficult to see what changes were made without some digging), we provide two patches 
which contain all of our modifications to these repos that can be applied with a single git command
(instructions are provided here).
We also believe this gives more credit and visability to the original authors.


# WGAN-GP

Our patch implements the functionality to train a model with our clipped methods, as well as produce
gradient histograms.

## Instructions

1. Download the reference codebase, and checkout a specific commit that we developed from:

    ```
    git clone git@github.com:w86763777/pytorch-gan-collections.git
    cd pytorch-gan-collections
    git checkout 860e90bd882eb5737103de55bcf0e977d6952343
    ```

2. Apply our patch, which contains code for gradient clipping and generating histograms, and setup 
    the virtual environment:

    ```bash
    git apply /PATH/TO/clipped_pytorch-gan-collections.patch

    # setup the virtualenv (wherever you desire) and activate
    virtualenv ~/wgan-venv
    source ~/wgan-venv/bin/activate

    # install deps
    pip install -U pip setuptools
    pip install -r requirements.txt
    pip install pytorch_gan_metrics --no-deps

    # if you have a newer version of torch and it doesn't automatically download cu113:
    # pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

    # directory required for cached data
    mkdir stats
    ```
    
    If you have issues with the `torch` version, just get a recent version of `torch` and `torchvision` and make sure it's compatible with your version of cuda.

    We only work with CIFAR, which will automatically be downloaded when you run an experiment. 
    You must however follow the instructions in the [original repo](https://github.com/w86763777/pytorch-gan-collections)
    to download [`cifar10.train.npz`](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC) to `./stats`.

3. You can run our experiments using the following command:

    ```bash
    python wgangp.py --arch=res32 --batch_size=64 --dataset=cifar10 --seed=0  --alpha=10 \
        --fid_cache=./stats/cifar10.train.npz  --loss=was --num_images=50000 \
        --record --sample_step=500 --sample_size=64 --total_steps=100000 --z_dim=128 \
        --logdir=./logs/wgangp \
        --opt=sgd \
        --lr_D=0.02 \
        --lr_G=0.02 \
        --clip_mode=value  \
        --clip_G=0.01 \
        --clip_D=0.01 \
        --desc='reproduce'
    ```

    The main parameters to experiment with are `--lr_G`/`--lr_D` (the generator and discriminator learning
    rates), `--clip-mode=(none|value|norm)`, and `--clip_G`/`--clip_D`, the clip parameters for
    the generator and discriminator respectively. `--opt=(sgd|extrasgd)` specifies whether to use
    regular SGD or the extragradient version of SGD.
    `--desc` is just a helpful user-defined string which is appended to the output log directory as 
    tag - this has no impact on the training or evaluation.

    The commands to reproduce the *"best models"* are:

    ```bash
    # vanilla SGD
    python wgangp.py --arch=res32 --batch_size=64 --dataset=cifar10 --seed=0  --alpha=10 \
        --fid_cache=./stats/cifar10.train.npz  --loss=was --num_images=50000 \
        --record --sample_step=500 --sample_size=64 --total_steps=100000 --z_dim=128 \
        --logdir=./logs/wgangp \
        --opt=sgd \
        --lr_D=0.0004 \
        --lr_G=0.0002 \
        --desc='sgda'
 
    # clipped-SGDA (norm)
    python wgangp.py --arch=res32 --batch_size=64 --dataset=cifar10 --seed=0  --alpha=10 \
        --fid_cache=./stats/cifar10.train.npz  --loss=was --num_images=50000 \
        --record --sample_step=500 --sample_size=64 --total_steps=100000 --z_dim=128 \
        --logdir=./logs/wgangp \
        --opt=sgd \
        --lr_D=0.2 \
        --lr_G=0.2 \
        --clip_mode=norm  \
        --clip_G=1 \
        --clip_D=1 \
        --desc='clipnorm-sgda'
 
    # clipped-SGDA (norm)
    python wgangp.py --arch=res32 --batch_size=64 --dataset=cifar10 --seed=0  --alpha=10 \
        --fid_cache=./stats/cifar10.train.npz  --loss=was --num_images=50000 \
        --record --sample_step=500 --sample_size=64 --total_steps=100000 --z_dim=128 \
        --logdir=./logs/wgangp \
        --opt=extrasgd \
        --lr_D=0.2 \
        --lr_G=0.2 \
        --clip_mode=norm  \
        --clip_G=1 \
        --clip_D=1 \
        --desc='clipnorm-seg'

    # clipped-SGDA (coordinate)
    python wgangp.py --arch=res32 --batch_size=64 --dataset=cifar10 --seed=0  --alpha=10 \
        --fid_cache=./stats/cifar10.train.npz  --loss=was --num_images=50000 \
        --record --sample_step=500 --sample_size=64 --total_steps=100000 --z_dim=128 \
        --logdir=./logs/wgangp \
        --opt=sgd \
        --lr_D=0.02 \
        --lr_G=0.02 \
        --clip_mode=value  \
        --clip_G=0.01 \
        --clip_D=0.01 \
        --desc='clipval-sgda'

    # clipped-SEG (coordinate)
    python wgangp.py --arch=res32 --batch_size=64 --dataset=cifar10 --seed=0  --alpha=10 \
        --fid_cache=./stats/cifar10.train.npz  --loss=was --num_images=50000 \
        --record --sample_step=500 --sample_size=64 --total_steps=100000 --z_dim=128 \
        --logdir=./logs/wgangp \
        --opt=extrasgd \
        --lr_D=0.02 \
        --lr_G=0.02 \
        --clip_mode=value  \
        --clip_G=0.01 \
        --clip_D=0.01 \
        --desc='clipval-seg'
    ```

6. If you wish to generate some gradient noise histograms, you need to compute the noise norms explicitly
    for whichever checkpoint you desire, eg:

    ```bash
        python wgangp.py --arch=res32 --batch_size=64 --dataset=cifar10 \
        --fid_cache=./stats/cifar10.train.npz --loss=was --record --sample_step=500 \
        --sample_size=20 --total_steps=1000 --logdir=./eval_runs/wgangp \
        --seed=0 --evaluate \
        --pretrain=./logs/wgangp/<EXPERIMENT ID>/model_xxxxxxxx.pt \
        --desc='eval'
    ```

    Don't forget to set the correct checkpoint for the `--pretrain` flag. The optimizer doesn't 
    matter as there are no aprameter updates.
    In the corresponding outdir, there will be a `results_norm.png` image which contains the 
    stochastic gradient noise histograms for the generator and discriminator (as well as the 
    raw generator and discriminator stochastic noise norms as npy files that you can load and 
    visualize yourself).


# StyleGAN2

Our patch provides the necessary functionality to reproduce our best trained model, as well as to generate
some of the noise histograms if desired.
Note that the nature of this code is highly experimental, and does not focus on providing a well engineered
implementation of our approach to the StyleGAN2 repo.

Ultimately, our implementation fundamentally amounts to using pytorches built-in gradient clipping 
functionality (after all gradient penalties are computed), and switching the optimizer from Adam to SGD.
We experimented with a number of other techniques, and the code provided here reflects that.

All of the original StyleGAN2 code is property of Nvidia; ensure you respect their license agreement. 

## Instructions

1. Download the official StyleGAN**3** Nvidia repository, and checkout a specific commit that we developed from:

    ```
    git clone git@github.com:NVlabs/stylegan3.git
    cd stylegan3
    git checkout 407db86e6fe432540a22515310188288687858fa
    ```

2. Apply our patch, which contains code for gradient clipping and generating histograms, and setup the conda
    environment:

    ```bash
    git apply /PATH/TO/clipped_stylegan3.patch
    conda env create -f environment.yml
    conda activate stylegan3
    ```

3. You must acquire the original FFHQ dataset (as instructed in the official Nvidia repository). 
    We then used Nvidia's provided `dataset_tool.py` script to resize the data to 128x128:

    ```bash
    python dataset_tool.py --source=/PATH/TO/ffhq/images1024x1024.zip \
        --dest=~/datasets/ffhq-128x128.zip \
        --resolution=128x128
    ```
    This will be the dataset used for all of our experiments.

4. You can reproduce our best model following the guidelines (make sure to set the appropriate path:
    for the `--resume` flag where necessary)

    ```bash
    # Run the first part for ~6000 kimgs (or until FID stops improving/worsens)
    python train.py --outdir=./logs/ --data=~/datasets/ffhq-128x128.zip --cfg=stylegan2 \
        --gpus=1 --batch=32 --gamma=0.1024 --map-depth=2 --cbase=16384 \
        --optimizer=sgd \
        --glr=0.35  \
        --dlr=0.35  \
        --clip-mode=value \
        --clip-gmax=0.0025 \
        --clip-dmax=0.0025 \
        --kimg=6000 \
        --desc="best_pt1"

    # Run the second part for ~3600 kimgs (or until FID stops improving/worsens)
    #   --> Ensure you set --resume=<model.pkl> correctly!
    python train.py --outdir=./logs/ --data=~/datasets/ffhq-128x128.zip --cfg=stylegan2 \
        --gpus=1 --batch=32 --gamma=0.1024 --map-depth=2 --cbase=16384 \
        --optimizer=sgd \
        --glr=0.035  \
        --dlr=0.035  \
        --clip-mode=value \
        --clip-gmax=0.0025 \
        --clip-dmax=0.0025 \
        --kimg=3600 \
        --resume=./logs/<EXPERIMENT_STRING>/network-snapshot-006000.pkl \
        --desc="best_pt2"

    # Run the final part for until FID stops improving/worsens
    #   --> Ensure you set --resume=<model.pkl> correctly!
    python train.py --outdir=./logs/ --data=~/datasets/ffhq-128x128.zip --cfg=stylegan2 \
        --gpus=1 --batch=32 --gamma=0.1024 --map-depth=2 --cbase=16384 \
        --optimizer=sgd \
        --glr=0.0035  
        --dlr=0.0035  \
        --clip-mode=value \
        --clip-gmax=0.0025 
        --clip-dmax=0.0025 \
        --kimg=8000 \
        --resume=./logs/<EXPERIMENT_STRING>/network-snapshot-003600.pkl \
        --desc="best_pt3"
    ```

    The main parameters to experiment with are `--glr`/`--dlr` (the generator and discriminator learning
    rates), `--clip-mode=(none|value|norm)`, and `--clip-gmax`/`--clip-dmax`, the clip parameters for
    the generator and discriminator respectively. `--kimg` specifes how many kimgs to train for, and
    `--desc` is just a helpful user-defined string which is appended to the output log directory as 
    tag - this has no impact on the training or evaluation.

6. If you wish to generate some gradient noise histograms, you need to compute the noise norms explicitly
    for whichever checkpoint you desire:

    ```bash
    python gradnoise_train.py --outdir=./eval_runs --cfg=stylegan2 --data=~/datasets/ffhq-128x128.zip \
        --gpus=1 --batch=32 --gamma=0.1024 --cbase=16384 --kimg=1000 --mode=both --metrics=none \
        --desc="eval" \
        --resume=./logs/<EXPERIMENT_STRING>/network-snapshot-XXXXXX.pkl
    ```

    Once again, don't forget to set the correct checkpoint for the `--resume` flag. 
    In the corresponding outdir, there will be a `results_norm.png` image which contains the 
    stochastic gradient noise histograms for the generator and discriminator (as well as the 
    raw generator and discriminator stochastic noise norms as npy files that you can load and 
    visualize yourself).


# License and References

If you want to use our code, please, cite our work:

> @misc{gorbunov2022clipped,
>       title={Clipped Stochastic Methods for Variational Inequalities with Heavy-Tailed Noise}, 
>       author={Eduard Gorbunov and Marina Danilova and David Dobre and Pavel Dvurechensky and Alexander Gasnikov and Gauthier Gidel},
>       year={2022},
>       eprint={2206.01095},
>       archivePrefix={arXiv},
>       primaryClass={math.OC}
> }

The StyleGAN2 codebase is code made available by NVIDIA with the following license:

> Copyright &copy; 2021, NVIDIA Corporation & affiliates. All rights reserved.
> Made available under the [Nvidia Source Code License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt).
