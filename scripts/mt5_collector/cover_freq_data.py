import pandas as pd
import os

# 加载1分钟数据
# 读取原始 CSV 文件

source_dir = '/Users/admin/.qlib/csv_data/forex_1min'  # 替换为你的文件路径
cover_freq = "D"


def load_symbols_data(directory: str):
    # 获取目录下的所有文件
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    # 读取每个文件并合并到一个 DataFrame 中

    for file in files:
        symbol = file.split('.')[0]  # 获取符号名称
        df = pd.read_csv(f"{source_dir}/{file}")
        df['date'] = pd.to_datetime(df['date'])  # 确保 'datetime' 列是 datetime 类型
        df.set_index('date', inplace=True)
        df_freq = cover_freq_data(df, cover_freq)

        # 保存5分钟数据
        df_freq.to_csv(f"/Users/admin/.qlib/csv_data/forex_day/{symbol}.csv")


def cover_freq_data(df, cover_freq: str):
    # 转换为5分钟数据
    df_freq = df.resample(cover_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return df_freq


if __name__ == '__main__':
    load_symbols_data(source_dir)
    print("转换完成")
