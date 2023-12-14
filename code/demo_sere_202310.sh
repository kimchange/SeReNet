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


python ./test.py --gpu 0 --savefolder $usingsavefolder --psfshift $usingpsfshift --model $usingmodel --AOstar 0 --resolution 101,1183,1183 --inp_size 273 --overlap 0 --config $usingconfig --savebitdepth 32 --startframe 0 --order 0 --savevolume 1 --codefolder $usingcodefolder --replacestr ${usingreplacestr}

# programName=${0##*/}
# cp $programName ./${usingreplacestr}
# cp ./test.py ./${usingreplacestr}

echo "结束测试......"