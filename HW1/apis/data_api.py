import pandas as pd 
import numpy as np  
from matplotlib import pyplot as plt
import requests as rq
from typing import List, Tuple
from io import BytesIO
import os 


base_url = "https://web.metro.taipei/RidershipPerStation/{}_cht.ods"
class MonthData:
    def __init__(self):
        self.month_date = []
    def add(self, new_month: pd.Series):
        self.month_date.append(new_month)
    def get_month(self, idx: int) -> pd.Series:
        return self.month_date[idx]
    def total_days(self, idx:int=-1) -> int:
        if idx == 0:
            return 0
        return sum([len(m) for m in self.month_date[:idx]])
    def __len__(self):
        return len(self.month_date)
    def get_full_list (self) -> pd.Series:
        return  pd.Series(pd.concat(self.month_date, axis=0).reset_index(drop=True))
    


def get_data(date_list: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, MonthData, pd.Series]:
    """
    An API to get the entry_data auto matically from the web
    Args:
        date_str (str): example "202501" for January 2025
    Returns:
        pd.DataFrame: A en)try_data frame containing the entry_data such as 
                        index: station names
                        columns: dates
                        values: ridership numbers
                        
    example:
        entry_data, date_data, avg = get_data(['202501','202502','202503'])
    """
    entry_data = pd.DataFrame()
    exit_data = pd.DataFrame()
    date_data = MonthData()
    for d in date_list:
        os.makedirs(os.path.join(os.getcwd(), 'HW1', 'data'), exist_ok=True)
        file_path = os.path.join(os.getcwd(), 'HW1', 'data', f"{d}_cht.ods")
        if os.path.exists(file_path):
            file_path = os.path.join(os.getcwd(), 'HW1', 'data', f"{d}_cht.ods")
            if entry_data.empty and exit_data.empty:
                entry_data = pd.read_excel(file_path, engine='odf', sheet_name=1)
                exit_data = pd.read_excel(file_path, engine='odf', sheet_name=0)
                date_data.add(pd.to_datetime(entry_data['　　　　車站日期']).rename(f'{d}').dt.strftime('%m-%d'))
            else:
                et_tmp = pd.read_excel(file_path, engine='odf', sheet_name=1)
                ex_tmp = pd.read_excel(file_path, engine='odf', sheet_name=0)
                entry_data = pd.concat([entry_data, et_tmp], axis=0)
                exit_data =  pd.concat([exit_data, ex_tmp], axis=0)
                
                date_data.add(pd.to_datetime(ex_tmp['　　　　車站日期']).rename(f'{d}').dt.strftime('%m-%d'))
        else:
            print(f"Downloading data for {d}")
            url = base_url.format(d)
            response = rq.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                file_tmp = BytesIO(response.content)
                if entry_data.empty and exit_data.empty:
                    entry_data = pd.read_excel(file_tmp, engine='odf', sheet_name=1)
                    file_tmp.seek(0)
                    exit_data = pd.read_excel(file_tmp, engine='odf', sheet_name=0)
                    date_data.add(pd.to_datetime(entry_data['　　　　車站日期']).rename(f'{d}').dt.strftime('%m-%d'))
                else:
                    et_tmp = pd.read_excel(file_tmp, engine='odf', sheet_name=1)
                    file_tmp.seek(0)
                    ex_tmp = pd.read_excel(file_tmp, engine='odf', sheet_name=0)
                    entry_data = pd.concat([entry_data, et_tmp], axis=0)
                    exit_data =  pd.concat([exit_data, ex_tmp], axis=0)
                    
                    date_data.add(pd.to_datetime(ex_tmp['　　　　車站日期']).rename(f'{d}').dt.strftime('%m-%d'))
                    
            else:
                print(f"Failed to download data for {d}. Status code: {response.status_code}")
    
    avg = entry_data.mean(axis=0)

    entry_data.drop(["　　　　車站日期"], axis=1, inplace=True)
    exit_data.drop(["　　　　車站日期"], axis=1, inplace=True)
    
    return entry_data, exit_data, date_data, avg

if __name__ == "__main__":
    entries, exits, time, avg = get_data(['202501','202502','202503'])
    print(entries)
    print(exits)
    print(time)
    print(avg)