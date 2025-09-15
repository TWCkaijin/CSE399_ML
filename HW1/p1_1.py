from .apis.data_api import get_data
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  
select_num = 4

def main():
    
    entry_data, exit_data, date_data = get_data(['202501','202502','202503'])
    

    random_month = 2 #np.random.randint(len(date_data))
    print(f"Randomly selected month: {random_month}")
    start_day = date_data.total_days(random_month)
    end_day = start_day + date_data.get_month(random_month).shape[0]
    print("Plotting data from day {} to day {}".format(start_day, end_day))

    r_v = np.random.choice(entry_data.keys(),(select_num,), replace=False)
    entry_data = entry_data.get_month(random_month)[list(r_v)]
    exit_data = exit_data.get_month(random_month)[list(r_v)]

    for entry_name in entry_data.keys():
        plt.plot(entry_data[entry_name], label=f'{entry_name}', alpha=0.3)

    entry_avg = np.average(entry_data[:], axis=1)
    entry_avg = pd.Series(entry_avg, index=entry_data.index)
    entry_avg.dropna(inplace=True)
    plt.plot(entry_avg[:], label='entry_avg', color='red', linewidth=3)


    dates = date_data.get_month(random_month).to_list()
    plt.xticks(range(end_day-start_day), dates, rotation=90)
    plt.legend(entry_data.keys().to_list()+['average'])
    plt.show()