import pandas as pd
import numpy as np
import math
import openpyxl
import sys

if len(sys.argv) < 2:
    print("Usage: Enter the file path prefix e.g. Chiang/10MM")
    exit()
pre_path = sys.argv[1]  # Chiang/10MM
save_name = pre_path + "_sen_stderr.csv"
excel_name = pre_path + "_sen_table.xlsx"
df = pd.read_csv(pre_path + ".csv")
init_th = df['THRESHOLD'][0]
pass_num = len(df.loc[df['THRESHOLD'] == init_th])
FP_MAX = 8


array = []
for i in range(1, pass_num+1):
    for j in range(0, FP_MAX):
        array.append([i, j, np.nan])


df_case_fp = pd.DataFrame(
    columns=['PASS', 'FP', 'SENSITIVITY'], data=array).set_index(['PASS', 'FP'])

for fp in range(FP_MAX):
    for i in range(1, pass_num+1):
        data_locate = df.loc[(df['FP'] == fp) & (df['FILENAME'] == i)]
        if data_locate.empty == False:
            df_case_fp['SENSITIVITY'][i][fp] = data_locate['SENSITIVITY'].max()

for i in range(1, pass_num+1):
    df_case_fp['SENSITIVITY'][i] = df_case_fp['SENSITIVITY'][i].interpolate()

df_case_fp = df_case_fp.reset_index().sort_values(
    by=['FP', 'PASS']).set_index(['FP', 'PASS'])
df_case_fp.to_excel(excel_name)

fp_sen_array = []
for fp in range(FP_MAX):
    sen_avg = df_case_fp['SENSITIVITY'][fp].mean()
    sen_std = df_case_fp['SENSITIVITY'][fp].std()
    sen_err = sen_std / math.sqrt(pass_num)
    sen_ci95_hi = sen_avg + 1.96*sen_err
    sen_ci95_lo = sen_avg - 1.96*sen_err
    fp_sen_array.append([fp, sen_avg,
                         sen_std, sen_err, sen_ci95_hi, sen_ci95_lo])
df_case_fp_err = pd.DataFrame(columns=[
                              'FP', 'SEN_AVG', 'SEN_STD', 'SEN_ERR', 'CI_95_HI', 'CI_95_LO'], data=fp_sen_array)
df_case_fp_err.to_csv(save_name)
