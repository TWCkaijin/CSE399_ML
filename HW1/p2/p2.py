from HW1.apis.data_api import get_data
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    entry_data, exit_data, date_data = get_data(['202501','202502','202503'])
    station = np.random.choice(entry_data.keys(),(1,), replace=False)
    
    plt.plot(entry_data.get_full_list()[station[0]], label='Raw Data', linewidth=1.5, color="red")

    d = np.random.randint(3,8)
    mov_avg = entry_data.get_full_list()[station[0]].rolling(window=d).mean()
    plt.plot(mov_avg, label=f'{d}-Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel(f'Entry of {station[0]}')
    plt.legend()
    plt.show()