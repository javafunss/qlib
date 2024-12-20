from qlib.data.dataset.loader import QlibDataLoader


class CustomForexDataLoader(QlibDataLoader):
    """Dataloader to get Alpha158"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config(
        config={
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "volume": {},
            "rolling": {},
            "ta_lib": {
                "windows": [14, 20],
                "indicators": {
                    "atr": {
                        "multiplier": 1
                    },
                    "boll": {
                        "k": 2
                    },
                    "rsi": {}
                }
            }
        }
    ):
        """create factors from config

        config = {
            'kbar': {}, # whether to use some hard-code kbar features
            'price': { # whether to use raw price features
                'windows': [0, 1, 2, 3, 4], # use price at n days ago
                'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': ['ROC', 'MA', 'STD'], # rolling operator to use
                #if include is None we will use default operators
                'exclude': ['RANK'], # rolling operator not to use
            },
            'ta_lib': { # whether to use ta-lib based features
                'windows': [14, 20], # ta-lib windows size
                'indicators': {
                    'atr': {
                        'multiplier': 1 # multiplier for ATR
                    },
                    'boll': {
                        'k': 2 # multiplier for BOLL bands
                    },
                    'rsi': {} # RSI configuration
                }
            }
        }
        """
        fields = []
        names = []
        include = None
        exclude = None

        def use(x):
            return x not in exclude and (include is None or x in include)

        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += [
                "KMID",  # 收盘价与开盘价的差值占开盘价的比例。这个指标反映了当天价格的变化幅度
                "KLEN",  # 最高价与最低价的差值占开盘价的比例。这个指标反映了当天价格的波动范围
                "KMID2",  # 收盘价与开盘价的差值占当天价格波动范围的比例。这个指标进一步细化了价格变化的幅度
                "KUP",  # 最高价与当天较高价（开盘价或收盘价）的差值占开盘价的比例。这个指标反映了价格在高位的波动情况
                "KUP2",  # 最高价与当天较高价（开盘价或收盘价）的差值占当天价格波动范围的比例。这个指标进一步细化了价格在高位的波动情况
                "KLOW",  # 当天较低价（开盘价或收盘价）与最低价的差值占开盘价的比例。这个指标反映了价格在低位的波动情况
                "KLOW2",  # 当天较低价（开盘价或收盘价）与最低价的差值占当天价格波动范围的比例。这个指标进一步细化了价格在低位的波动情况
                "KSFT",  # 两倍收盘价减去最高价和最低价的差值占开盘价的比例。这个指标综合考虑了收盘价、最高价和最低价的关系
                "KSFT2",  # 两倍收盘价减去最高价和最低价的差值占当天价格波动范围的比例。这个指标进一步细化了收盘价、最高价和最低价的关系
            ]

        if "price" in config:
            # price 特征 生成不同时间窗口的价格比值，例如 $OPEN0 表示当天开盘价与收盘价的比值，$OPEN1 表示前一天开盘价与当天收盘价的比值
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature:
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + str(d) for d in windows]

        if "volume" in config:
            # volume 特征  生成不同时间窗口的成交量比值，例如 VOLUME0 表示当天成交量与当天成交量的比值，VOLUME1 表示前一天成交量与当天成交量的比值
            windows = config["volume"].get("windows", range(5))
            fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]

        if "rolling" in config:
            # rolling 特征 生成基于滚动窗口的各种指标，例如 ROC5 表示 5 天的收益率，MA10 表示 10 天的移动平均线
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])

            if use("ROC"):
                # 过去 d 天的收盘价相对于当前收盘价的变化率。这个指标反映了价格在指定时间窗口内的变化速度
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC%d" % d for d in windows]
            if use("MA"):
                # 过去 d 天的收盘价的移动平均值。这个指标平滑了价格数据，帮助识别趋势
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA%d" % d for d in windows]
            if use("STD"):
                # 过去 d 天的收盘价的标准差。这个指标反映了价格在指定时间窗口内的波动程度
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD%d" % d for d in windows]
            if use("BETA"):
                # 过去 d 天的收盘价的线性回归斜率。这个指标反映了价格的趋势方向和强度
                fields += ["Slope($close, %d)/$close" % d for d in windows]
                names += ["BETA%d" % d for d in windows]
            if use("RSQR"):
                # 过去 d 天的收盘价的 R-squared 值。这个指标反映了线性回归模型对数据的拟合程度
                fields += ["Rsquare($close, %d)" % d for d in windows]
                names += ["RSQR%d" % d for d in windows]
            if use("RESI"):
                # 过去 d 天的收盘价的残差。这个指标反映了实际价格与线性回归预测价格之间的差异
                fields += ["Resi($close, %d)/$close" % d for d in windows]
                names += ["RESI%d" % d for d in windows]
            if use("MAX"):
                # 过去 d 天的最高价。这个指标反映了价格在指定时间窗口内的最高水平
                fields += ["Max($high, %d)/$close" % d for d in windows]
                names += ["MAX%d" % d for d in windows]
            if use("LOW"):
                # 过去 d 天的最低价。这个指标反映了价格在指定时间窗口内的最低水平
                fields += ["Min($low, %d)/$close" % d for d in windows]
                names += ["MIN%d" % d for d in windows]
            if use("QTLU"):
                # 过去 d 天的收盘价的上四分位数。这个指标反映了价格在指定时间窗口内的较高水平
                fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
                names += ["QTLU%d" % d for d in windows]
            if use("QTLD"):
                # 过去 d 天的收盘价的下四分位数。这个指标反映了价格在指定时间窗口内的较低水平
                fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
                names += ["QTLD%d" % d for d in windows]
            if use("RANK"):
                # 过去 d 天的收盘价的排名。这个指标反映了当前价格在历史价格中的相对位置
                fields += ["Rank($close, %d)" % d for d in windows]
                names += ["RANK%d" % d for d in windows]
            if use("RSV"):
                # 过去 d 天的收盘价相对于最高价和最低价的位置。这个指标反映了价格在指定时间窗口内的相对位置
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV%d" % d for d in windows]
            if use("IMAX"):
                # 过去 d 天的最高价出现的索引位置。这个指标反映了最高价在时间窗口内的出现时间
                fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
                names += ["IMAX%d" % d for d in windows]
            if use("IMIN"):
                # 过去 d 天的最低价出现的索引位置。这个指标反映了最低价在时间窗口内的出现时间
                fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
                names += ["IMIN%d" % d for d in windows]
            if use("IMXD"):
                # 过去 d 天的最高价和最低价出现时间的差值。这个指标反映了价格在时间窗口内的波动周期
                fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
                names += ["IMXD%d" % d for d in windows]
            if use("CORR"):
                # 过去 d 天的收盘价与对数化后的成交量之间的相关系数。这个指标反映了价格和成交量之间的关系
                fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
                names += ["CORR%d" % d for d in windows]
            if use("CORD"):
                # 过去 d 天的收益率与对数化后的成交量变化之间的相关系数。这个指标反映了价格变化和成交量变化之间的关系
                fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
                names += ["CORD%d" % d for d in windows]
            if use("CNTP"):
                # 过去 d 天内价格上涨的天数占比。这个指标反映了价格在指定时间窗口内的上涨概率。
                fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTP%d" % d for d in windows]
            if use("CNTN"):
                # 过去 d 天内价格下跌的天数占比。这个指标反映了价格在指定时间窗口内的下跌概率。
                fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTN%d" % d for d in windows]
            if use("CNTD"):
                # 过去 d 天内价格上涨天数与下跌天数的差值。这个指标反映了价格在指定时间窗口内的涨跌平衡情况
                fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
                names += ["CNTD%d" % d for d in windows]
            if use("SUMP"):
                # 过去 d 天内价格上涨部分的总和占价格变化绝对值总和的比例。这个指标反映了价格上涨的强度
                fields += [
                    "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMP%d" % d for d in windows]
            if use("SUMN"):
                # 过去 d 天内价格下跌部分的总和占价格变化绝对值总和的比例。这个指标反映了价格下跌的强度
                fields += [
                    "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMN%d" % d for d in windows]
            if use("SUMD"):
                # 过去 d 天内价格上涨部分的总和与下跌部分的总和的差值。这个指标反映了价格在指定时间窗口内的涨跌净额
                fields += [
                    "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                    "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["SUMD%d" % d for d in windows]
            if use("VMA"):
                # 过去 d 天的成交量的移动平均值。这个指标平滑了成交量数据，帮助识别成交量的趋势
                fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VMA%d" % d for d in windows]
            if use("VSTD"):
                # 过去 d 天的成交量的标准差。这个指标反映了成交量在指定时间窗口内的波动程度
                fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VSTD%d" % d for d in windows]
            if use("WVMA"):
                # 过去 d 天的加权成交量的移动平均值。这个指标反映了成交量在价格变化中的权重
                fields += [
                    "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["WVMA%d" % d for d in windows]
            if use("VSUMP"):
                # 过去 d 天内成交量增加部分的总和占成交量变化绝对值总和的比例。这个指标反映了成交量增加的强度
                fields += [
                    "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMP%d" % d for d in windows]
            if use("VSUMN"):
                # 过去 d 天内成交量减少部分的总和占成交量变化绝对值总和的比例。这个指标反映了成交量减少的强度
                fields += [
                    "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMN%d" % d for d in windows]
            if use("VSUMD"):
                # 过去 d 天内成交量增加部分的总和与减少部分的总和的差值。这个指标反映了成交量在指定时间窗口内的增减净额
                fields += [
                    "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["VSUMD%d" % d for d in windows]

        if "ta-lib" in config:
            windows = config["ta_lib"].get("windows", [5, 10, 20, 30])
            include = config["ta_lib"].get("include", None)
            exclude = config["ta_lib"].get("exclude", [])
            indicators = config["ta_lib"].get("indicators", {})

            for window in windows:
                if use("atr"):
                    multiplier = indicators.get("atr", {}).get("multiplier", 1)
                    tr = "Max($high-$low, Abs($high-Ref($close, 1)), Abs($low-Ref($close, 1)))"
                    atr = f"Mean({tr}, {window})"
                    fields.extend([tr, atr])
                    names.extend([f"TR_{window}", f"ATR_{window}"])

                if use("boll"):
                    k = indicators.get("boll", {}).get("k", 2)
                    mid_band = f"Mean($close, {window})"
                    upper_band = f"{mid_band} + {k} * Std($close, {window})"
                    lower_band = f"{mid_band} - {k} * Std($close, {window})"
                    fields.extend([mid_band, upper_band, lower_band])
                    names.extend([f"BOLL_MID_{window}", f"BOLL_UPPER_{window}", f"BOLL_LOWER_{window}"])

                if use("rsi"):
                    gain = "Greater($close-Ref($close, 1), 0)"
                    loss = "Greater(Ref($close, 1)-$close, 0)"
                    avg_gain = f"Mean({gain}, {window})"
                    avg_loss = f"Mean({loss}, {window})"
                    rs = f"{avg_gain} / ({avg_loss} + 1e-12)"
                    rsi = f"100 - 100 / (1 + {rs})"
                    fields.extend([gain, loss, avg_gain, avg_loss, rs, rsi])
                    names.extend([f"GAIN_{window}", f"LOSS_{window}", f"AVG_GAIN_{window}", f"AVG_LOSS_{window}", f"RS_{window}", f"RSI_{window}"])

        return fields, names
