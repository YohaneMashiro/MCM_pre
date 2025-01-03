import pandas as pd

# 读取 Excel 文件
file_path = r'D:\document\MCM\week1\data\data.xlsx'
data = pd.read_excel(file_path)

# 将数据导出为 CSV
data.to_csv(r'D:\document\MCM\week1\data\data.csv', index=False)
