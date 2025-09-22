import pandas as pd 
import numpy as np  
from matplotlib import pyplot as plt
import requests as rq
from typing import List, Tuple
from io import BytesIO
import os 


base_url = "https://web.metro.taipei/RidershipPerStation/{}_cht.ods"
class Data:
    def __init__(self):
        self.data = []
    def add(self, new_month):
        self.data.append(new_month)
    def get_month(self, idx: int):
        return self.data[idx]
    def total_days(self, idx:int=-1) -> int:
        if idx == 0:
            return 0
        return sum([len(m) for m in self.data[:idx]])
    def __len__(self):
        return len(self.data)
    def get_full_list (self):
        return  pd.DataFrame(pd.concat(self.data, axis=0).reset_index(drop=True))
    def empty(self) -> bool:
        return len(self.data) == 0
    def keys(self):
        if self.empty():
            return []
        return self.data[0].columns
    


def get_data(date_list: List[str]) -> Tuple[Data, Data, Data]:
    """
    An API to get the entry_data automatically from the web
    Args:
        date_str (str): example "202501" for January 2025
    Returns:
        pd.DataFrame: A entry_data frame containing the entry_data such as 
                        index: station names
                        columns: dates
                        values: ridership numbers
                        
    example:
        entry_data, date_data, avg = get_data(['202501','202502','202503'])
    """
    entry_data = Data()
    exit_data = Data()
    date_data = Data()
    for d in date_list:
        os.makedirs(os.path.join(os.getcwd(), 'HW1', 'data'), exist_ok=True)
        file_path = os.path.join(os.getcwd(), 'HW1', 'data', f"{d}_cht.ods")
        if os.path.exists(file_path):
            print(f"Loading data for {d} from local file")
            et_tmp = pd.read_excel(file_path, engine='odf', sheet_name=1)
            ex_tmp = pd.read_excel(file_path, engine='odf', sheet_name=0)
            
            date_data.add(pd.to_datetime(ex_tmp['　　　　車站日期']).rename(f'{d}').dt.strftime('%m-%d'))
            et_tmp.drop(["　　　　車站日期"], axis=1, inplace=True)
            ex_tmp.drop(["　　　　車站日期"], axis=1, inplace=True)
            entry_data.add(et_tmp)
            exit_data.add(ex_tmp)
            
        else:
            print(f"Downloading data for {d}")
            url = base_url.format(d)
            response = rq.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                file_tmp = BytesIO(response.content)
                et_tmp = pd.read_excel(file_tmp, engine='odf', sheet_name=1)
                file_tmp.seek(0)
                ex_tmp = pd.read_excel(file_tmp, engine='odf', sheet_name=0)
                
                date_data.add(pd.to_datetime(ex_tmp['　　　　車站日期']).rename(f'{d}').dt.strftime('%m-%d'))
                et_tmp.drop(["　　　　車站日期"], axis=1, inplace=True)
                ex_tmp.drop(["　　　　車站日期"], axis=1, inplace=True)
                
                entry_data.add(et_tmp)
                exit_data.add(ex_tmp)
            else:
                print(f"Failed to download data for {d}. Status code: {response.status_code}")
    
    return entry_data, exit_data, date_data

if __name__ == "__main__":
    entries, exits, time= get_data(['202501','202502','202503'])
