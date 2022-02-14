import miceforest as mf
import warnings
import os
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np


exl = pd.read_excel('clinical_info.xlsx', sheet_name=None, index_col=None, header=0)
df_concat_origin = pd.concat([exl['AS'], exl['non-AS'], exl['health']], join='inner')
var_sch = {
    'Disease Duration': ['Gender',
                         'Age',
                         'ESR',
                         'CRP',
                         'HLA-B27'],
    'ESR': ['Gender',
            'Age',
            'Disease Duration',
            'CRP',
            'HLA-B27'],
    'CRP': ['Gender',
            'Age',
            'Disease Duration',
            'ESR',
            'HLA-B27']
}
iter_count = 0
for epoch in range(200):
    print(f'epoch = {epoch}')
    iter_count +=1
    kds = mf.MultipleImputedKernel(
        df_concat_origin,
        variable_schema=var_sch,
        datasets=1,
        save_all_iterations=False,
        random_state=iter_count
    )
    kds.mice(5)
    df_concat = kds.complete_data(dataset=0)
    while df_concat.isnull().sum().sum() > 84:
        iter_count += 1
        kds = mf.MultipleImputedKernel(
            df_concat_origin,
            variable_schema=var_sch,
            datasets=1,
            random_state=iter_count
        )
        kds.mice()
        df_concat = kds.complete_data(dataset=0)

        print(f'self.iter_count = {iter_count}')
    df_concat.set_index('Number', inplace=True)

    os.makedirs('./log_excels', exist_ok=True)
    writer = pd.ExcelWriter(f'./log_excels/{epoch}.xlsx')
    df_concat.to_excel(writer)
    writer.save()
