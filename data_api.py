import pandas as pd 
import numpy as np  
from matplotlib import pyplot as plt
import requests as rq
from typing import List, Tuple
from io import BytesIO
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  

base_url = "https://web.metro.taipei/RidershipPerStation/{}_cht.ods"

def get_data(date_list: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    An API to get the data auto matically from the web
    Args:
        date_str (str): example "202501" for January 2025
    Returns:
        pd.DataFrame: A data frame containing the data such as 
                        index: station names
                        columns: dates
                        values: ridership numbers
    """
    data = pd.DataFrame()
    for d in date_list:
        print(f"Downloading data for {d}")
        url = base_url.format(d)
        response = rq.get(url)
        if response.status_code == 200:
            file_tmp = BytesIO(response.content)
            if data.empty:
                data = pd.read_excel(file_tmp, engine='odf')
            else:
                data = pd.concat([data, pd.read_excel(file_tmp, engine='odf')], axis=0)
        else:
            print(f"Failed to download data for {d}. Status code: {response.status_code}")
        print(f"{d} downloaded , shape: {data.shape}")
    avg = data.mean(axis=0)
    time_list = pd.to_datetime(data['　　　　車站日期'])
    time_series = time_list.dt.strftime('%m-%d')
    data.drop(["　　　　車站日期"], axis=1, inplace=True)
    return data, time_series, avg


x, y, z = get_data(['202501','202502','202503'])
print(x)
print(y)
print(z)