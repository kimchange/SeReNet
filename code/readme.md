# create ananconda environment

conda create -n serenet python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install h5py imageio tifffile tqdm pyyaml tensorboard tensorboardX
conda install scipy matplotlib scikit-image

# calculate psf and psfshift

cd ~/SeReNet/psf/psfcalc/

matlab 
main_computePSF_serenet

# generate synthetic bubtubbead data using for training

cd ~/SeReNet/data/
genBubtubbead_imaging_urlfm_20231204


cd ~/SeReNet/psf/

python get_psfshift.py


cd ~/SeReNet/code/

python train_serenet.py