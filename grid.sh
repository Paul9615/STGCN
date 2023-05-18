# !/bin/bash

# epochs=() -> keep
cheb=(1 2 3 4 5)
step=(10 11 12 13 14 15)
gamma=(0.95 0.90 0.85 0.80)
de=(0.005 0.01 0.015)

# cheb=(1)
# gamma=(0.95 0.90)

for c in "${cheb[@]}"; do
    for z in "${step[@]}"; do
        for g in "${gamma[@]}"; do
            for d in "${de[@]}"; do
                loss=$(python3 main.py --cases=parameter --dataset=pemsd7-m --order=True --Ks=$c --step_size=$z --gamma=$g --weight_decay_rate=$d)
                result+=($c $z $g $d $loss)
            done
        done
    done
done

# for c in "${cheb[@]}"; do
#     for g in "${gamma[@]}"; do
#         # loss=$(python3 main.py --cases=parameter --dataset=pemsd7-m --order=True --Ks=$c --step_size=$z --gamma=$g --weight_decay_rate=$d)
#         loss=$(python3 main.py --cases=parameter --dataset=pemsd7-m --order=True --Ks=$c --gamma=$g)
#         # result+=($c $z $g $d $loss)
#         result+=($c $g $loss)
#     done
# done

# best_score=($(echo "${result[@]:4:5}" | sort -n | head -1))
# echo ${result[@]}
best_score=($(echo "${result[@]:4:5}" | sort -n | head -1))
bast_para=($(echo "${result[@]}" | grep "$best_score"))

echo "The best hyper parameter is: cheb kernel ${bast_para[0]} step size ${bast_para[1]} gamma ${bast_para[2]} decay weight ${bast_para[3]}"
# echo "The best hyper parameter is: cheb kernel ${bast_para[0]} gamma ${bast_para[1]}"

echo "test using found parameter"
# python3 main.py --cases=tuned_parameter --dataset=pemsd7-m --Ks=${bast_para[0]} --gamma=${bast_para[1]}
python3 main.py --cases=tuned_parameter --dataset=pemsd7-m --Ks=${bast_para[0]} --step_size=${bast_para[1]} --gamma=${bast_para[2]} --weight_decay_rate=${bast_para[3]}