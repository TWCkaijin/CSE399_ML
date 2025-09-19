from HW1.apis.data_api import get_data
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():

    entry_data, exit_data, date_data = get_data(['202501','202502','202503'])
    station = np.random.choice(entry_data.keys(),(1,), replace=False)[0]
    print(entry_data.data[0][station])
    
    fig, ax = plt.subplots(1,2,figsize=(10, 6))
    ax[0].boxplot(entry_data.get_full_list()[station], positions=[1], labels=[f'entry'])
    ax[1].hist(entry_data.get_full_list()[station], bins=30, alpha=0.5, label=f'Month entry')
    
    ax[0].boxplot(exit_data.get_full_list()[station], positions=[2], labels=[f'exit'])
    ax[1].hist(exit_data.get_full_list()[station], bins=30, alpha=0.5, label=f'Month exit')
    
    plt.suptitle(f'{station}')
    ax[0].set_title(f'Boxplot of {station}')
    ax[1].set_title(f'Histogram of {station}')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()