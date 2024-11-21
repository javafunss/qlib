import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

"""1. 原始函数核心逻辑说明
主要功能
dtfx_algo_zones 函数的主要功能是计算并绘制价格波动中的支撑和阻力区域（即“Zones”），这些区域基于斐波那契回撤水平（Fibonacci Retracement Levels）。具体来说，它通过识别价格的高低点（Swing Points）来确定这些区域，并根据用户设置的参数来绘制相应的斐波那契水平线和区域。

核心逻辑
输入参数：

data: 包含 OHLC（开盘价、最高价、最低价、收盘价）数据的 DataFrame。
各种配置参数，如 structure_len（结构长度）、zone_disp_num（显示最后几个区域）、disp_all（是否显示所有区域）、zone_filter（显示哪些类型的区域）、no_overlap（是否清理重叠的水平线）等。
计算最高价和最低价：

使用滚动窗口计算每根 K 线的最高价和最低价。
方向判断：

dir 变量用于跟踪当前的价格趋势方向（上升或下降）。
当价格突破最近的最高价时，dir 设置为 -1（下降趋势）。
当价格跌破最近的最低价时，dir 设置为 1（上升趋势）。
识别高低点：

t 和 b 分别存储最近的高点和低点。
当识别到新的高点或低点时，更新 t 和 b 的值。
计算斐波那契水平：

使用 get_fibs 函数计算斐波那契回撤水平。
根据当前的趋势方向（上升或下降），计算相应的斐波那契水平。
绘制区域和水平线：

使用 plot_zones 和 plot_fibs 函数绘制支撑和阻力区域以及斐波那契水平线。
根据用户设置的参数，决定是否显示所有区域或仅显示最后几个区域。
清理重叠的水平线：

如果 no_overlap 参数为 True，则清理重叠的斐波那契水平线，确保每个区域的水平线不重叠。
2. 根据 dtfx_algo_zones 构建交易信号的说明
交易信号生成逻辑
识别高低点：

通过识别价格的高低点（Swing Points），确定当前的趋势方向。
高点（Swing High）表示潜在的阻力位，低点（Swing Low）表示潜在的支撑位。
计算斐波那契水平：

根据识别到的高低点，计算斐波那契回撤水平。
这些水平线可以作为潜在的支撑和阻力位。
生成交易信号：

买入信号：当价格从下方突破某个斐波那契水平线时，生成买入信号。
卖出信号：当价格从上方跌破某个斐波那契水平线时，生成卖出信号。
具体实现
在 on_bar 方法中，调用 dtfx_algo_zones 函数计算并返回支撑和阻力区域及其斐波那契水平线。然后根据这些水平线生成交易信号。
    """


def linestyle(style_str):
    styles = {
        "___": "-",
        "- - -": "--",
        ". . .": ":"
    }
    return styles.get(style_str, "-")


def get_fibs(top, bot, direction, fl1, fl2, fl3, fl4, fl5):
    rng = top - bot if direction == 1 else bot - top
    anchor = bot if direction == 1 else top
    fib1 = anchor + (rng * fl1)
    fib2 = anchor + (rng * fl2)
    fib3 = anchor + (rng * fl3)
    fib4 = anchor + (rng * fl4)
    fib5 = anchor + (rng * fl5)
    return fib1, fib2, fib3, fib4, fib5


def plot_fibs(ax, t, b, direction, fib1, fib2, fib3, fib4, fib5, f1_style, f2_style, f3_style, f4_style, f5_style, f1_bull_color, f2_bull_color, f3_bull_color, f4_bull_color, f5_bull_color, f1_bear_color, f2_bear_color, f3_bear_color, f4_bear_color, f5_bear_color):
    colors = {
        1: (f1_bull_color, f2_bull_color, f3_bull_color, f4_bull_color, f5_bull_color),
        -1: (f1_bear_color, f2_bear_color, f3_bear_color, f4_bear_color, f5_bear_color)
    }
    for fib, style, color in zip([fib1, fib2, fib3, fib4, fib5], [f1_style, f2_style, f3_style, f4_style, f5_style], colors[direction]):
        if not np.isnan(fib):
            ax.axhline(y=fib, color=color, linestyle=style)


def plot_zones(ax, zones, bull_zone_color, bear_zone_color):
    for zone in zones:
        if zone['direction'] == 1:
            color = bull_zone_color
        else:
            color = bear_zone_color
        rect = Rectangle((zone['start_bar'], zone['top']), zone['end_bar'] - zone['start_bar'], zone['bottom'] - zone['top'], facecolor=color, edgecolor='none', alpha=0.5)
        ax.add_patch(rect)


def dtfx_algo_zones(data, structure_len=10, zone_disp_num=1, disp_all=True, zone_filter="Both", no_overlap=True, f1_tog=False, f2_tog=True, f3_tog=True, f4_tog=True, f5_tog=False, f1_lvl=0, f2_lvl=0.3, f3_lvl=0.5, f4_lvl=0.7, f5_lvl=1, f1_style=". . .", f2_style="- - -", f3_style="___", f4_style="- - -", f5_style=". . .", f1_bull_color="#089981", f2_bull_color="#089981", f3_bull_color="#089981", f4_bull_color="#089981", f5_bull_color="#089981", f1_bear_color="#f23645", f2_bear_color="#f23645", f3_bear_color="#f23645", f4_bear_color="#f23645", f5_bear_color="#f23645", structure_color="gray", bull_zone_color="#089981", bear_zone_color="#f23645"):
    data['upper'] = data['high'].rolling(window=structure_len).max()
    data['lower'] = data['low'].rolling(window=structure_len).min()

    dir = 0
    top = np.nan
    btm = np.nan
    t = {'price': np.nan, 'bar': np.nan}
    b = {'price': np.nan, 'bar': np.nan}
    bos_up_check = False
    bos_down_check = False
    last_bot = "NA"
    last_top = "NA"
    zones = []
    lvls = []

    for i in range(len(data)):
        if dir >= 0 and data.loc[i, 'high'] > data.loc[i, 'upper']:
            dir = -1
            top = data.loc[i, 'high']

        if dir <= 0 and data.loc[i, 'low'] < data.loc[i, 'lower']:
            dir = 1
            btm = data.loc[i, 'low']

        top_conf = not np.isnan(top)
        bot_conf = not np.isnan(btm)

        if top_conf:
            t = {'price': top, 'bar': i - structure_len}
            bos_up_check = True

        if bot_conf:
            b = {'price': btm, 'bar': i - structure_len}
            bos_down_check = True

        HH = top_conf and t['price'] > data.loc[i - 1, 'high']
        HL = bot_conf and b['price'] > data.loc[i - 1, 'low']
        LH = top_conf and t['price'] < data.loc[i - 1, 'high']
        LL = bot_conf and b['price'] < data.loc[i - 1, 'low']

        last_top = "HH" if HH else "LH" if LH else last_top
        last_bot = "LL" if LL else "HL" if HL else last_bot

        t_dir = 0
        choch_up = (data.loc[i, 'close'] > t['price']) and (data.loc[i, 'high'] == data.loc[i, 'upper']) and t_dir <= 0
        choch_down = (data.loc[i, 'close'] < b['price']) and (data.loc[i, 'low'] == data.loc[i, 'lower']) and t_dir >= 0

        if choch_up:
            t_dir = 1
        if choch_down:
            t_dir = -1

        bos_up = (data.loc[i, 'close'] > t['price']) and bos_up_check and t_dir >= 0
        bos_down = (data.loc[i, 'close'] < b['price']) and bos_down_check and t_dir <= 0

        mss_up = bos_up or choch_up
        mss_down = bos_down or choch_down

        if mss_up and zone_filter != "Bearish Only":
            _top = t['price']
            _bot = data.loc[i, 'lower'] if dir == -1 else b['price']
            fib1, fib2, fib3, fib4, fib5 = get_fibs(_top, _bot, 1, f1_lvl, f2_lvl, f3_lvl, f4_lvl, f5_lvl)
            live_zone = {'start_bar': t['bar'], 'end_bar': i, 'top': _top, 'bottom': _bot, 'direction': 1}
            live_lvls = {
                'f1': fib1 if f1_tog else np.nan,
                'f2': fib2 if f2_tog else np.nan,
                'f3': fib3 if f3_tog else np.nan,
                'f4': fib4 if f4_tog else np.nan,
                'f5': fib5 if f5_tog else np.nan
            }
            zones.append(live_zone)
            lvls.append(live_lvls)

        if mss_down and zone_filter != "Bullish Only":
            _top = data.loc[i, 'upper'] if dir == 1 else t['price']
            _bot = b['price']
            fib1, fib2, fib3, fib4, fib5 = get_fibs(_top, _bot, -1, f1_lvl, f2_lvl, f3_lvl, f4_lvl, f5_lvl)
            live_zone = {'start_bar': b['bar'], 'end_bar': i, 'top': _top, 'bottom': _bot, 'direction': -1}
            live_lvls = {
                'f1': fib1 if f1_tog else np.nan,
                'f2': fib2 if f2_tog else np.nan,
                'f3': fib3 if f3_tog else np.nan,
                'f4': fib4 if f4_tog else np.nan,
                'f5': fib5 if f5_tog else np.nan
            }
            zones.append(live_zone)
            lvls.append(live_lvls)

        if len(zones) > zone_disp_num and not disp_all:
            zones.pop(0)
            lvls.pop(0)

        if len(lvls) > 1 and no_overlap:
            last_zone = zones[-2]
            last_lvl = lvls[-2]
            if last_lvl['f1'] > live_lvls['f1']:
                last_lvl['f1'] = max(live_zone['start_bar'], last_zone['end_bar'])
                last_lvl['f2'] = max(live_zone['start_bar'], last_zone['end_bar'])
                last_lvl['f3'] = max(live_zone['start_bar'], last_zone['end_bar'])
                last_lvl['f4'] = max(live_zone['start_bar'], last_zone['end_bar'])
                last_lvl['f5'] = max(live_zone['start_bar'], last_zone['end_bar'])

                live_lvls['f1'] = max(live_zone['start_bar'], last_zone['end_bar'])
                live_lvls['f2'] = max(live_zone['start_bar'], last_zone['end_bar'])
                live_lvls['f3'] = max(live_zone['start_bar'], last_zone['end_bar'])
                live_lvls['f4'] = max(live_zone['start_bar'], last_zone['end_bar'])
                live_lvls['f5'] = max(live_zone['start_bar'], last_zone['end_bar'])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['close'], label='Close Price')
    plot_zones(ax, zones, bull_zone_color, bear_zone_color)
    for lvl in lvls:
        plot_fibs(ax, t, b, 1, lvl['f1'], lvl['f2'], lvl['f3'], lvl['f4'], lvl['f5'], f1_style, f2_style, f3_style, f4_style, f5_style, f1_bull_color, f2_bull_color, f3_bull_color, f4_bull_color, f5_bull_color, f1_bear_color, f2_bear_color, f3_bear_color, f4_bear_color, f5_bear_color)
    ax.legend()
    plt.show()


# 示例数据
data = pd.DataFrame({
    'open': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'high': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'low': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'close': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
})

dtfx_algo_zones(data)
