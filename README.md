# SeReNet

Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light field microscopy ([paper)](https://www.nature.com/articles/s41592-025-02698-z)

# Overview

We propose a physics-driven self-supervised reconstruction network (SeReNet) for light-field microscopy (LFM) and scanning LFM (sLFM) by incorporating angular point spread function (PSF) priors, achieving spatially-uniform near-diffraction-limit resolution at millisecond processing speed. Rather than brute-force full-supervised learning, SeReNet adopts a self-supervised scheme, and leverages the regularized wave-optics physical priors and guide the reliable convergence to a uniform high-resolution volume without ground truth required. Furthermore, we develop an axially finetuned version of SeReNet, by additionally introducing a small number of simulated data pair priors. A series of optimizations (NLL-MPG loss, preDAO and TW-Net) enable SeReNet highly robust to optical perturbations and sample motions, broadly generalizable and free from missing cone problem.
More details please refer to the companion paper where this method first occurred [[paper](https://www.nature.com/articles/s41592-025-02698-z)]. Next, we will guide you step by step to implement our method.

# Environments

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

Our repo is build mainly using PyTorch, so installing torch has higher priority, you can refer to torch [PyTorch](https://pytorch.org/) guide to install torch based on your machine and driver.

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
cd ~/SeReNet/trainingset/bubtub_Nnum13/dataset_generation_code
matlab -nodesktop
genBubtub_Nnum13_size1989
```

then the synthetic bubtub dataset can be found at `~/SeReNet/trainingdata/bubtub_Nnum13/x3_synthetic/`, and the folder name was written into example config.

### Train SeReNet using bubtub dataset

```
cd ~/SeReNet/save/month_202505/config_tag/code
python train_serenet.py
```

After about 16 hours on a single NVIDIA-A100-SXM4 GPU, the SeReNet model will converage and be saved at `~/SeReNet/pth/serenet_pth/epoch-800.pth`

### Test the network

If you want to try SeReNet, you can run

```
cd ~/SeReNet/demo
python test.py --model ./pth/serenet_pth/epoch-800.pth --inputfile ./data_3x3/L929_cell.tif
```

Then the spatial-angular images at `~/SeReNet/demo/data_3x3/L929_cell.tif` will be reconstructed into a 3D volume at `~/SeReNet/demo/data_3x3_recon/L929_cell_serenet.tif`

## Demo of axially finetuned SeReNet

### Train F-SeReNet

F-SeReNet (axially finetuned SeReNet) requires a pretrained SeReNet as an initial model, for example F-SeReNet can be trained based on the previously trained model at `~/SeReNet/pth/serenet_pth/epoch-800.pth`.  Synthetic bubtub 3D volume data are involved in F-SeReNet training.

```
cd ~/SeReNet/save/month_202505/config_tag/code
python train_fserenet.py
```

After about 6 hours, the F-SeReNet model will converage and be saved at `~/SeReNet/save/202505/fserenet_Nnum13_bubtub/epoch-800.pth`

### Test the network

run

```
cd ~/SeReNet/demo/
python test.py --model ./pth/fserenet_pth/epoch-800.pth --inputfile ./data_3x3/L929_cell.tif
```

If the used device cuda memory is less than 40GB, the patched reconstruction with overlap method can be used by running

```
cd ~/SeReNet/demo
python test.py --model ./pth/fserenet_pth/epoch-800.pth --inp_size 126 --overlap 15 --inputfile ./data_3x3/L929_cell.tif
```

The reconstruction of spatial-angular images at `~/SeReNet/demo/data_3x3/L929_cell.tif` can be found at `~/SeReNet/demo/data_3x3/L929_cell_fserenet.tif`

## Demo of TW-Net

### Test the pretrained network

```
cd ~/SeReNet/twnet/code
python test.py --model ../pth/twnet_pth/epoch-800.pth
```

# Results

A mitochondria labelled L929 cell (TOM20-GFP) was captured by sLFM. The spatial-angular measurements and the counterpart after TW-Net are shown in the left. The right part shows the results of SeReNet and axially finetuned SeReNet in the form of maximum intensity projections, which were obtained from the models pre-trained with the synthetic bubtub dataset. Scale bars, 10 μm. For more results and further analysis, please refer to the companion paper where this method first occurred [[paper]](unavailable now).

![results.png](images/results.png)

# Comparing methods

To demonstrate the comparisons to SOTA methods, this section summarizes a case where the performances of SeReNet, axially improved SeReNetm, [VCD-Net](https://github.com/xinDW/VCD-Net), [HyLFM-Net](https://github.com/kreshuklab/hylfm-net) and [RL-Net](https://github.com/MeatyPlus/Richardson-Lucy-Net) are evaluated on the same training and test datasets. All codes are placed together in `~/SeReNet/save/month_202505/config_tag/codes`, and have been rewritten using PyTorch. We removed unnecessary python site-package dependencies, creating a friendly-use architecture for the community. Some network structures were modified to adapt to scanning light-field data. Hyperparameters were kept as faithful as possible to the original implementations, with several modifications due to differences in angular views and spatial sampling rates.

Previously trained parameters can be found at [link](https://drive.google.com/drive/folders/1jBqrK7dsvIqnuw8YSYRxfucRmNnGl2pq?usp=sharing), one can download all of them and put them into `~/SeReNet/demo/pth/` folder.

## Using SeReNet

Trained with bubtub dataset and reconstruct brain slice data

```
cd ~/SeReNet/save/month_202505/config_tag/code
python train_serenet.py --config ./configs/train-serenet/serenet_config.yaml 
cd ~/SeReNet/demo/
python test.py --model ../save/202505/serenet_Nnum13_bubtub/epoch-800.pth
```

Or use the network model trained previously in `~/SeReNet/demo/pth/serenet_pth/epoch-800.pth` file path

```
cd ~/SeReNet/demo/
python test.py --model ./pth/serenet_pth/epoch-800.pth
```

## Using axially improved SeReNet

Trained with bubtub dataset and reconstruct brain slice data **after training the serenet**.

```
cd ~/SeReNet/save/month_202505/config_tag/code
python train_fserenet.py --config ./configs/train-fserenet/fserenet_Nnum13_bubtub.yaml 
cd ~/SeReNet/demo
python test.py --model ../save/202505/fserenet_Nnum13_bubtub/epoch-800.pth
```

Or use the network model trained previously in `~/SeReNet/demo/pth/fserenet_pth/epoch-800.pth` file path.

```
cd ~/SeReNet/demo
python test.py --model ./pth/fserenet_pth/epoch-800.pth --inp_size 126 --overlap 15
```

## Using VCD-Net

Trained with bubtub dataset and reconstruct brain slice data

```
cd ~/SeReNet/save/month_202505/config_tag/code
python train_supervised.py --config ./configs/train-supervised/vcdnet_bubtub.yaml 
cd ~/SeReNet/demo/
python test_supervised.py --model ../save/202505/vcdnet_bubtub/epoch-800.pth
```

Or use the network model trained previously in `~/SeReNet/demo/pth/vcdnet_pth/epoch-800.pth` file path

```
cd ~/SeReNet/demo/
python test_supervised.py --model ./pth/vcdnet_pth/epoch-800.pth
```

## Using HyLFM-Net

Trained with bubtub dataset and reconstruct brain slice data

```
cd ~/SeReNet/save/month_202505/config_tag/code
python train_supervised.py --config ./configs/train-supervised/hylfmnet_bubtub.yaml
cd ~/SeReNet/demo/
python test_supervised.py --model ../save/202505/hylfmnet_bubtub/epoch-800.pth
```

Or use the network model trained previously in `~/SeReNet/demo/pth/hylfmnet_pth/epoch-800.pth` file path

```
cd ~/SeReNet/demo/
python test_supervised.py --model ./pth/hylfmnet_pth/epoch-800.pth
```

## Using RL-Net (RLN)

RL-Net is a supervised network, not specifically designed for LFM. To adapt RL-Net for sLFM images, we conducted preprocessing. Spatial-angular views of sLFM were transformed into a volume through upsampling and digital refocusing, as the input of RL-Net. The corresponding high-resolution volume is used as the target, to train the RL-Net.

Trained with bubtub dataset and reconstruct brain slice data

```
cd ~/SeReNet/save/month_202505/config_tag/code
python train_rlnet.py --config ./configs/train-rlnet/rlnet_bubtub.yaml
cd ~/SeReNet/demo
python test_rlnet.py --model ../save/202505/rlnet_bubtub/epoch-800.pth
```

Or use the network model trained previously in `~/SeReNet/demo/pth/rlnet_pth/epoch-800.pth` file path

```
cd ~/SeReNet/demo/
python test_rlnet.py --model ./pth/rlnet_pth/epoch-800.pth
```

## Using Vs-Net (VsLFM)

VsNet is a supervised network to upsamle the input low resolution light fields into high resolution one.

Trained with bubtub dataset and reconstruct brain slice data

```
cd ~/SeReNet/save/month_202505/config_tag/code
python train_vsnet.py --config ./configs/train-vsnet/vsnet_Nnum13_datamix.yaml
cd ~/SeReNet/demo
python test_vsnet.py --model ../save/202505/vsnet_Nnum13_bubtub/epoch-800.pth
```

Or use the network model trained previously in `~/SeReNet/demo/pth/vsnet_pth/vsnet.pth` file path

```
cd ~/SeReNet/demo/
python test_vsnet.py --model ./pth/vsnet_pth/vsnet.pth
```

# Citation

If you use this code and relevant data, please cite the corresponding paper where original methods appeared:
[[paper](https://www.nature.com/articles/s41592-025-02698-z)]

```
Lu, Z., Jin, M., Chen, S. et al. Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light-field microscopy. Nat Methods (2025). https://doi.org/10.1038/s41592-025-02698-z
```

# Correspondence

Should you have any questions regarding this project and the corresponding results, please contact Zhi Lu (luzhi@tsinghua.edu.cn).
