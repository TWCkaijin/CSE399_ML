from apis.data_api import get_data
import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  
select_num = 4
entry_data, exit_data, date_data, avg = get_data(['202501','202502',"202503","202504","202505","202506"])
r_v = np.random.choice(entry_data.keys(),(select_num,), replace=False)
entry_data = entry_data[list(r_v)]
exit_data = exit_data[list(r_v)]

random_month = np.random.randint(len(date_data))
print(f"Randomly selected month: {random_month}")
start_day = date_data.total_days(random_month)
print(start_day)
end_day = start_day + date_data.get_month(random_month).shape[0]
print(end_day)


for entry_name in entry_data.keys():
    plt.plot(entry_data[entry_name][start_day:end_day], label=f'{entry_name}', alpha=0.3)

entry_avg = np.average(entry_data[:], axis=1)
entry_avg = pd.Series(entry_avg, index=entry_data.index)
entry_avg.dropna(inplace=True)
plt.plot(entry_avg[start_day:end_day], label='entry_avg', color='red', linewidth=3)


dates = date_data.get_month(random_month).to_list()
plt.xticks(range(end_day-start_day), dates, rotation=90)
plt.legend(entry_data.keys().to_list()+['average'])
plt.show()