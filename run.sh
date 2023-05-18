# !/bin/bash
data=$1

if [ $data = "pem" ]
then
python3 main.py --cases=init_pemsbay_ks5 --gpu=6 --dataset=pems-bay --Ks=5
python3 main.py --cases=init_pemsbay_ks6 --gpu=6 --dataset=pems-bay --Ks=6
python3 main.py --cases=init_pemsbay_ks7 --gpu=6 --dataset=pems-bay --Ks=7

elif [ $data = "metr" ]
then
python3 main.py --cases=init_metrla_ks5 --gpu=2 --dataset=metr-la --Ks=5
python3 main.py --cases=init_metrla_ks6 --gpu=2 --dataset=metr-la --Ks=6
python3 main.py --cases=init_metrla_ks7 --gpu=2 --dataset=metr-la --Ks=7

elif [ $data = "p7" ]
then
python3 main.py --cases=init_pemsd7m_ks5 --gpu=4 --dataset=pemsd7-m --Ks=5 --n_pred=9
python3 main.py --cases=init_pemsd7m_ks6 --gpu=4 --dataset=pemsd7-m --Ks=6 --n_pred=9
python3 main.py --cases=init_pemsd7m_ks7 --gpu=4 --dataset=pemsd7-m --Ks=7 --n_pred=9

fi