import pandas as pd
import math
import sys

if len(sys.argv) < 2:
    print("Usage: Enter the file path prefix e.g. Chiang/10MM")
    exit()
pre_path = sys.argv[1]  # Chiang/10MM
#csv_full_path = "all_threshold_"+csv_id+".csv"
csv_full_path = pre_path + ".csv"
save_path = pre_path + "_fix_th_stderr.csv"
extract_path = pre_path + "_ext_fix_th_stderr.csv"


df = pd.read_csv(csv_full_path)
th_table = df.loc[:, 'THRESHOLD'].drop_duplicates()
res = []
for th in th_table:
    single_th_df = df.loc[df['THRESHOLD'] == th]
    sum_df = single_th_df.sum()
    mean_df = single_th_df.mean()
    std_df = single_th_df.std()

    sen_avg = sum_df['TP']/(sum_df['TP']+sum_df['FN'])
    sen_std = std_df['SENSITIVITY']
    sen_err = sen_std / math.sqrt(len(single_th_df))
    sen_ci95_hi = sen_avg + 1.96*sen_err
    sen_ci95_lo = sen_avg - 1.96*sen_err

    fp_avg = sum_df['FP']/len(single_th_df)
    fp_std = std_df['FP']
    fp_err = fp_std / math.sqrt(len(single_th_df))
    fp_ci95_hi = fp_avg + 1.96*fp_err
    fp_ci95_lo = fp_avg - 1.96*fp_err

    res.append([th, sen_avg, sen_std, sen_err, sen_ci95_hi, sen_ci95_lo,
                fp_avg, fp_std, fp_err, fp_ci95_hi, fp_ci95_lo
                ])
    #add_result(th,sen_avg,fp_avg,sen_std, sen_err,sen_ci95_hi,sen_ci95_lo)

df_result = pd.DataFrame(columns=['THRESHOLD', 'SEN_AVG', 'SEN_STD', 'SEN_ERR', 'SEN_CI95_HI',
                                  'SEN_CI95_LO', 'FP_AVG', 'FP_STD', 'FP_ERR', 'FP_CI95_HI', 'FP_CI95_LO'], data=res)
df_result.to_csv(save_path, index=0)


target_sen = [70, 75, 80, 85, 90, 95, 98, 99, 100]
pointer = 0
res.sort(key=lambda x: [x[1], x[6]])
extract = []
for l in res:
    if pointer >= len(target_sen):
        break
    print(l[0], l[1], l[6], target_sen[pointer]/100)
    if l[1] >= target_sen[pointer]/100:
        extract.append(l)
        pointer += 1

df_result_extract = pd.DataFrame(columns=['THRESHOLD', 'SEN_AVG', 'SEN_STD', 'SEN_ERR', 'SEN_CI95_HI',
                                          'SEN_CI95_LO', 'FP_AVG', 'FP_STD', 'FP_ERR', 'FP_CI95_HI', 'FP_CI95_LO'], data=extract)
df_result_extract.to_csv(extract_path, index=0)
