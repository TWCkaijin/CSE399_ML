from HW1.apis.data_api import get_data
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():

    entry_data, exit_data, date_data = get_data(['202501','202502','202503'])
    select_row = 4
    station = np.random.choice(entry_data.keys(),(select_row**2,2), replace=False)
    fig, ax = plt.subplots(4,4,figsize=(12, 8))
    for i, st in enumerate(station):
        ax[i//4][i%4].scatter(entry_data.get_full_list()[st[0]], exit_data.get_full_list()[st[1]], alpha=0.5)
        ax[i//4][i%4].set_title(f'{st[0]} / {st[1]}')
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    main()