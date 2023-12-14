#!/usr/bin/env bash
 
echo "开始测试......"



usingnametag=202311/_train_serenet_bubtubbead_aber0IpsfResizex13_x3_23
usingnametag=202312/_train_serenet_bubtubbead_aber0IpsfResizex13_x3_04
usingnametag=202312/_train_serenetsf_bubtubbead_aber0IpsfResizex13_x3_04
usingcodefolder=../save/$usingnametag/code/
usingmodel=$usingcodefolder/../epoch-800.pth
usingconfig=$usingcodefolder/../config.yaml
usingpsfshift=../psf/Ideal_PhazeSpacePSF_M63_NA1.4_zmin-10u_zmax10u_zspacing0.2u_psfshift_49views.pt
usingreplacestr=serenet/$usingnametag/
usingsavefolder=../data/${usingreplacestr}mito/


# python ./test.py --gpu 7 --savefolder $usingsavefolder --psfshift $usingpsfshift --model $usingmodel --AOstar 0 --resolution 101,1183,1183 --inp_size 273 --overlap 0 --config $usingconfig --savebitdepth 32 --startframe 0 --order 0 --savevolume 1 --codefolder $usingcodefolder --replacestr ${usingreplacestr}


# python ./test.py --gpu 7 --savefolder ../data/010/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_010/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/011/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_011/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/012/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_012/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/013/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_013/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/014/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_014/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/015/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_015/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/016/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_016/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/017/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_017/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/018/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_018/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/019/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_019/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/020/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_020/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/010/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_010/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/011/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_011/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/012/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_012/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/013/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_013/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/014/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_014/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/015/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_015/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/016/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_016/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/017/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_017/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/018/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_018/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/019/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_019/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/020/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_020/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./


# python ./test.py --gpu 7 --savefolder ../data/subbg/000/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_000/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/001/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_001/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/002/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_002/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/003/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_003/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/004/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_004/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/005/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_005/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/006/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_006/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/007/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_007/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/008/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_008/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# python ./test.py --gpu 7 --savefolder ../data/subbg/009/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202312/_train_twnet_amiba_x3_009/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

python ./test.py --gpu 6 --savefolder ../data/subbg/202313/_train_twnet_amiba_x3_020/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202313/_train_twnet_amiba_x3_020/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

python ./test.py --gpu 6 --savefolder ../data/subbg/202313/_train_twnet_amiba_x3_020-submean_deleted_threshold0.2mean_deleted/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202313/_train_twnet_amiba_x3_020-submean_deleted_threshold0.2mean_deleted/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

python ./test.py --gpu 6 --savefolder ../data/subbg/202313/_train_twnet_amiba_x3_020-0.2mean_deleted/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202313/_train_twnet_amiba_x3_020-0.2mean_deleted/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

python ./test.py --gpu 6 --savefolder ../data/subbg/202313/_train_twnet_amiba_x3_015-submean_deleted_threshold0.15mean_deleted/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202313/_train_twnet_amiba_x3_015-submean_deleted_threshold0.15mean_deleted/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

python ./test.py --gpu 6 --savefolder ../data/subbg/202313/_train_twnet_amiba_x3_010-submean_deleted_threshold0.10mean_deleted/ --model /home/slfm/as13000/VDF-LFM/SeReNet/twnet/save/202313/_train_twnet_amiba_x3_010-submean_deleted_threshold0.10mean_deleted/epoch-800.pth --savebitdepth 16 --startframe 0 --order 0 --savevolume 1 --codefolder ./

# programName=${0##*/}
# cp $programName ./${usingreplacestr}
# cp ./test.py ./${usingreplacestr}

echo "结束测试......"