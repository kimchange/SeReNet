# SeReNet

Self-supervised reconstruction network for microscopy light field images.

# Overview

We propose a physical model-driven self-supervised reconstruction network (SeReNet) for light-field microscopy (LFM) and scanning LFM (sLFM) by incorporating angular point spread function (PSF) priors, achieving spatially-uniform near-diffraction-limit resolution at millisecond processing speed. Rather than brute-force full-supervised learning, SeReNet adopts a self-supervised scheme, and leverages the regularized wave-optics physical priors and guide the reliable convergence to a uniform high-resolution volume without ground truth required. Furthermore, we develop an axially finetuned version of SeReNet, by additionally introducing a small number of simulated data pair priors. A series of optimizations (NLL-MPG loss, preDAO and TW-Net) enable SeReNet highly robust to optical perturbations and sample motions, broadly generalizable and free from missing cone problem.
More details please refer to the companion paper where this method first occurred [[paper]](unavailable now). Next, we will guide you step by step to implement our method.

# environments

## Recomented System Configuration

* a NVIDIA-A100-SXM4 / NVIDIA-RTX3090 gpu or better
* 128GB RAM
* 1TB disk space
* Scanning Light field microscopy captured data

## Preparation

### Download SeReNet source code

Download our code using

```
cd ~
git clone https://github.com/kimchange/SeReNet.git
```

### Create SeReNet ananconda environment

```
conda create -n serenet python=3.9
conda activate serenet
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install h5py imageio tifffile tqdm pyyaml scipy matplotlib scikit-image tensorboard tensorboardX
```

### Obtain phase-space PSF and its center of mass "psfshift"

<!-- ```
cd ~/SeReNet/psf/psfcalc/

matlab
main_computePSF_serenet
``` -->

<!-- cd ~/SeReNet/psf/

python get_psfshift.py -->

The PSF file can be download from [here](https://drive.google.com/drive/folders/1FieOk-oLh0xGwOxP5IyXufbuSrLxrgvu?usp=drive_link), then put it into `~/SeReNet/psf/`

# DEMO

## Demo of SeReNet

### Train SeReNet

If you want to train SeReNet with synthetic or experimental light field data, then you can make a config like `~/SeReNet/code/configs/train-serenet/serenet_config.yaml`, the example training pipeline follows:

### Generate synthetic bubtub dataset used for training

```
cd ~/SeReNet/data/
genBubtubbead_imaging_urlfm_20231204
```

then the synthetic bubtub dataset can be found at `~/SeReNet/trainingdata/bubtub/x3_synthetic/`, and the folder name was written into example config.

### Train SeReNet using bubtub dataset

```
cd ~/SeReNet/code/
python train_serenet.py
```

After about 16 hours on a single NVIDIA-A100-SXM4 GPU, the SeReNet model will converage and be saved at `~/SeReNet/pth/serenet_pth/epoch-800.pth`

### Test the network

If you want to try SeReNet, you can run

```
matlab -nodesktop
cd ~/SeReNet/code
python test.py --model ../pth/serenet_pth/epoch-800.pth
```

Then the spatial-angular images at `~/SeReNet/data/demo_WDF_input.tif` will be reconstructed into a 3D volume at `~/SeReNet/data/demo_WDF_input_serenet.tif`

## Demo of axially finetuned SeReNet

### Train F-SeReNet

F-SeReNet (axially finetuned SeReNet) requires a pretrained SeReNet as an initial model, for example F-SeReNet can be trained based on the previously trained model at `~/SeReNet/pth/serenet_pth/epoch-800.pth`.  Synthetic bubtub 3D volume data are involved in F-SeReNet training.

```
cd ~/SeReNet/code
python train_fserenet.py
```

After about 6 hours, the F-SeReNet model will converage and be saved at `~/SeReNet/pth/fserenet_pth/epoch-800.pth`

### Test the network

run

```
cd ~/SeReNet/code
python test.py --model ../pth/fserenet_pth/epoch-800.pth
```

The reconstruction of spatial-angular images at `~/SeReNet/data/demo_WDF_input.tif` can be found at `~/SeReNet/data/demo_WDF_input_fserenet.tif`

## Demo of TW-Net

### Test the pretrained network

```
cd ~/SeReNet/twnet/code
python test.py --model ../pth/twnet_pth/epoch-800.pth
```

# Results

A mitochondria labelled L929 cell (TOM20-GFP) was captured by sLFM. The spatial-angular measurements and the counterpart after TW-Net are shown in the left. The right part shows the results of SeReNet and axially finetuned SeReNet in the form of maximum intensity projections, which were obtained from the models pre-trained with the synthetic bubtub dataset. Scale bars, 10 μm. For more results and further analysis, please refer to the companion paper where this method first occurred [[paper]](unavailable now).

![results.png](images/results.png)

# Citation

If you use this code and relevant data, please cite the corresponding paper where original methods appeared:
[[paper]](unavailable now)

# Correspondence

Should you have any questions regarding this project and the corresponding results, please contact Zhi Lu (luzhi@tsinghua.edu.cn).
