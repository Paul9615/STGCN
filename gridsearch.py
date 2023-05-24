import os 
import subprocess as sb 
from multiprocessing import Process as pro
from multiprocessing import Pool

import pandas as pd
import re

def search(paras): # gpu_num: int, cheb_ker: int
    val_res=pd.DataFrame(columns=['cheb','step', 'gamma', 'decay','val'])
    step = [i for i in range(10, 16)]
    gamma = [0.80, 0.85, 0.90, 0.95]
    de = [0.005, 0.01, 0.015]

    for s in step:
        for g in gamma:
            for d in de:
                print(f'the parameters: cheb kernel: {paras[1]} step size: {s} gamma: {g} decay rate: {d}')
                val_com = f"python3 main.py --gpu={paras[0]} --cases=parameter --dataset=pemsd7-m --order=True --Ks={paras[1]} --step_size={s} --gamma={g} --weight_decay_rate={d}"
                val = sb.check_output(val_com, shell=True)
                sub_res = {'cheb': paras[1], 'step':s, 'gamma':g,'decay':d,'val':val}
                val_res = val_res.append(sub_res, ignore_index=True)

    val_res.to_csv(f'./tuning/cheb{paras[1]}.csv')

def extract_float(x):
    match = re.search(r'(\d+\.\d+)', x)
    if match:
        return match.group()
    else:
        return None

def grid_search():
    ch1 = pd.read_csv('./tuning/cheb1.csv', index_col=0)
    ch2 = pd.read_csv('./tuning/cheb2.csv', index_col=0)
    ch3 = pd.read_csv('./tuning/cheb3.csv', index_col=0)
    ch4 = pd.read_csv('./tuning/cheb4.csv', index_col=0)
    ch5 = pd.read_csv('./tuning/cheb5.csv', index_col=0)

    res = pd.concat([ch1, ch2, ch3, ch4, ch5], ignore_index=True)
        
    res['val'] = res['val'].apply(lambda x: extract_float(x)).astype('float64')
    
    print(f'the result of all process \n {res.head()}')
    print('====================================================')
    print(res.tail())
    print('====================================================')

    sorted_res = res.sort_values('val')

    print(f'sorted result \n {sorted_res.head()}')
    print('====================================================')

    best_para = sorted_res.head(1)
    
    best_cheb=best_para['cheb'].item()
    best_cheb=best_para['step'].item() 
    best_gamma=best_para['gamma'].item() 
    best_decay=best_para['decay'].item()

    best_para_list = [best_cheb, best_cheb, best_gamma, best_decay]
    print(f'best parameter: cheb {best_para_list[0]} | step size {best_para_list[1]} | gamma {best_para_list[2]} | decay {best_para_list[3]}') 
    
    return best_para_list

def test(best_para):
    test_com = f"python3 main.py --cases=tune_test --dataset=pemsd7-m --Ks={best_para[0]} --step_size={best_para[1]} --gamma={best_para[2]} --weight_decay_rate={best_para[3]}"
    test = sb.check_output(test_com, shell=True)
    

if __name__ == "__main__":
    # cheb1 = pro(target=search, args=(0, 1))
    # cheb2 = pro(target=search, args=(1, 2))
    # cheb3 = pro(target=search, args=(2, 3))
    # cheb4 = pro(target=search, args=(3, 4))
    # cheb5 = pro(target=search, args=(4, 5))
    # cheb1.start(); cheb1.join()
    # cheb2.start(); cheb2.join()
    # cheb3.start(); cheb3.join()
    # cheb4.start(); cheb4.join()
    # cheb5.start(); cheb5.join()
    
    try:
        os.mkdir('./tuning')
    except: 
        pass
    
    
    
    done = False    
    gpu_ker = [[0,1], [1,2], [2,3], [3,4], [4,5]]
    pool = Pool(processes=5)
    pool.map(search, gpu_ker)
    pool.close()
    pool.join()
    done = True
    
    if done == True:
        best_para = grid_search()
        test(best_para)
    
