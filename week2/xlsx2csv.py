import pandas as pd

# 读取 Excel 文件
file_path = r'D:\document\MCM\week2\第二周小测数据\2021_MCM_Problem_C_Data\2021MCMProblemC_DataSet.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# 将数据导出为 CSV
data.to_csv(r'D:\document\MCM\week2\第二周小测数据\2021_MCM_Problem_C_Data\2021MCMProblemC_DataSet.csv', index=False)