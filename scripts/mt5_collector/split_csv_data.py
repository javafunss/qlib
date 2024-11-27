import pandas as pd

# 读取原始 CSV 文件
input_file = '/Users/admin/.qlib/csv_data/mt5_data/vnpy.bar_data.csv'  # 替换为你的文件路径
df = pd.read_csv(input_file)

# 定义需要保留的列名映射
column_mapping = {
    'symbol': 'symbol',
    'datetime': 'date',
    'open_price': 'open',
    'high_price': 'high',
    'low_price': 'low',
    'close_price': 'close',
    'volume': 'volume'
}

# 添加固定值为 1 的 factor 列
df['factor'] = 1

# 按 symbol 分组并保存每个符号的数据到单独的 CSV 文件
for symbol, group in df.groupby('symbol'):
    # 选择需要的列并重命名
    subset = group[list(column_mapping.keys()) + ['factor']].rename(columns=column_mapping)

    # 保存到新的 CSV 文件
    output_file = f'/Users/admin/.qlib/csv_data/forex/{symbol}.csv'
    subset.to_csv(output_file, index=False)
    print(f'Saved {output_file}')
