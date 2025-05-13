I optimized the project structure to make it more convenient to develop new ideas based on previous works.
You can see that this folder is "save", and it is organized as:

```text
📦 SeReNet (root_dir of your workspace)
├── 📂 psf
│        ├── 📄 psf_Nnum13.mat
│        └── 📄 psf_Nnum13_psfshift_49views.pt
│        └── 📄 psf_Nnum15_psfshift_81views.pt
├── 📂 save
│        ├── 📂 202503
│        └── 📂 202504
│        └── 📂 202505
│                     ├── 📂 model_config_your-tag-here
│                     └── 📂 serenet_12-ganloss
│                     └── 📂 serenet_12v3-NLLPG
│                                  ├── 📄 config.yaml
│                                  └── 📄 epoch-800.pt
│                                  └── 📄 log.txt
│                                  └── 📂 code
│                                               ├── 📂 configs
│                                               └── 📂 models
│                                               └── 📂 utils
│                                               ├── 📂 datasets
│                                                            ├── 📄 \__init__.py
│                                                            └── 📄 image_folder.py
│                                                            └── 📄 wrappers.py
│                                               └── 📄 train.py
│                                               └── 📄 utils.py
├── 📂 demo
│        ├── 📂 data_Nnum13_3x3
│        └── 📂 data_Nnum15_1x1
│        └── 📄 test_vsnet.py
│        └── 📄 test_serenet.py

├── 📂 your_captured_data
│        ├── 📂 data_Nnum13_3x3_20250513_slices
│        └── 📂 data_Nnum15_1x1_20250514_animals
│        └── 📄 test_vsnet.py
│        └── 📄 test_serenet.py
├── 📂 trainingset
│        ├── 📂 bubtub_Nnum13
│        └── 📂 bubtub_Nnum15
│        └── 📂 datamix_Nnum15
│        └── 📂 synthetic_Nnum13
│                     ├── 📂 x3_synthetic
│                     └── 📂 data_generation_code
│                     └── 📂 GT_synthetic
│                                  ├── 🔬 group001.tif
│                                  ├── 🔬 group002.tif
├── 📂 twnet
└── 📂 somethingelse
```

you can see that almost all training code are in 📂 save folder, and every time we want to start a training, change your dir to where 📄 train.py is, for example:

```
cd ~/SeReNet/save/202505/serenet_12v3-NLLMPG/code/
python train_serenet.py --config ./configs/train-serenet/serenet_Nnum13_bubtub.yaml --name 202505/ --tag 13v1-newtry --gpu 2 
```

then an backup of this folder will be automatically saved at

```
~/SeReNet/save/202505/serenet_Nnum13_bubtub_13v1-newtry/code
```

the training results, logs, intermediate files will be output to

```
~/SeReNet/save/202505/serenet_Nnum13_bubtub_13v1-newtry/
```

this can be more convenient to do something with code version control, especially when git server network not stable, and at the same time, this way will not occupy many disk space.


To be more specific, for example,  `~/SeReNet/save/202505/serenet_Nnum13_bubtub_13v1-newtry/code/configs` stores your hyper_parameters, training data, which model to use...

`~/SeReNet/save/202505/serenet_Nnum13_bubtub_13v1-newtry/code/datasets/image_folders.py` controls how your project interact with hardware disk storage

after load data from disk, `~/SeReNet/save/202505/serenet_Nnum13_bubtub_13v1-newtry/code/datasets/wrapper.py` controls how to feed (preprocess) the raw data into your network

after data preprocessing, `~/SeReNet/save/202505/serenet_Nnum13_bubtub_13v1-newtry/code/models/serenet.py` is the network choosed by `train_serenet.py` according to `configs`


As you can see, I have combined these methods: 

SeReNet, VsNet, HyLFM-Net, VCD-Net, RL-Net all in this one structure. Wish this can be helpful to developers.
