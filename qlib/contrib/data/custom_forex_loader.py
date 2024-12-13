import pandas as pd
import talib
from qlib.data.dataset.loader import QlibDataLoader


class CustomForexDataLoader(QlibDataLoader):
    """
    自定义外汇因子 DataLoader，支持动态配置特征计算窗口与操作，并集成 ta-lib 技术指标。
    """

    @staticmethod
    def get_feature_config(
        config={
            "price": {
                "windows": [0],  # 原始价格特征的窗口
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],  # 选择的价格字段
            },
            "rolling": {
                "windows": [5, 10, 20],  # 滚动窗口大小
                "include": ["MA", "STD", "ROC"],  # 包含的滚动操作
                "exclude": ["RANK"],  # 排除的滚动操作
            },
            "talib": {
                "indicators": ["RSI", "MACD", "ATR"],  # 要计算的技术指标
                "params": {"RSI": {"timeperiod": 14}},  # 技术指标参数
            },
        }
    ):
        """
        根据用户配置生成特征计算规则。
        """
        feature_config = {}

        # 添加价格特征配置
        if "price" in config:
            price_cfg = config["price"]
            feature_config["price"] = {
                "windows": price_cfg.get("windows", [0]),
                "feature": price_cfg.get("feature", ["OPEN", "HIGH", "LOW", "CLOSE"]),
            }

        # 添加滚动特征配置
        if "rolling" in config:
            rolling_cfg = config["rolling"]
            feature_config["rolling"] = {
                "windows": rolling_cfg.get("windows", [5, 10, 20]),
                "include": rolling_cfg.get("include", None),
                "exclude": rolling_cfg.get("exclude", None),
            }

        # 添加 ta-lib 指标配置
        if "talib" in config:
            talib_cfg = config["talib"]
            feature_config["talib"] = {
                "indicators": talib_cfg.get("indicators", ["RSI", "MACD", "ATR"]),
                "params": talib_cfg.get("params", {}),
            }

        return feature_config

    def __init__(self, feature_config=None, **kwargs):
        """
        初始化自定义 DataLoader，支持用户配置特征计算。
        """
        super().__init__(**kwargs)
        self.feature_config = self.get_feature_config(feature_config)

    def _compute_talib_features(self, df):
        """
        使用 ta-lib 计算技术指标。
        """
        talib_config = self.feature_config.get("talib", {})
        indicators = talib_config.get("indicators", [])
        params = talib_config.get("params", {})

        for indicator in indicators:
            if indicator == "RSI":
                timeperiod = params.get("RSI", {}).get("timeperiod", 14)
                df[f"RSI_{timeperiod}"] = talib.RSI(df["$close"], timeperiod=timeperiod)

            elif indicator == "MACD":
                fastperiod = params.get("MACD", {}).get("fastperiod", 12)
                slowperiod = params.get("MACD", {}).get("slowperiod", 26)
                signalperiod = params.get("MACD", {}).get("signalperiod", 9)
                macd, macdsignal, macdhist = talib.MACD(
                    df["$close"], fastperiod, slowperiod, signalperiod
                )
                df[f"MACD"] = macd
                df[f"MACD_signal"] = macdsignal

            elif indicator == "ATR":
                timeperiod = params.get("ATR", {}).get("timeperiod", 14)
                df[f"ATR_{timeperiod}"] = talib.ATR(
                    df["$high"], df["$low"], df["$close"], timeperiod
                )

        return df

    def _process_data(self, df):
        """
        数据处理流程，支持动态计算特征。
        """
        # 提取配置
        price_config = self.feature_config.get("price", {})
        rolling_config = self.feature_config.get("rolling", {})

        # ============================
        # 基础价格特征处理
        # ============================
        price_windows = price_config.get("windows", [0])
        price_features = price_config.get("feature", ["OPEN", "HIGH", "LOW", "CLOSE"])
        for window in price_windows:
            for feature in price_features:
                # 创建时间延迟特征
                df[f"{feature}_lag{window}"] = df[f"${feature.lower()}"].shift(window)

        # ============================
        # 滚动窗口特征处理
        # ============================
        rolling_windows = rolling_config.get("windows", [5, 10, 20])
        include_ops = rolling_config.get("include", ["MA", "STD", "ROC"])
        exclude_ops = rolling_config.get("exclude", [])

        for window in rolling_windows:
            if "MA" in include_ops and "MA" not in exclude_ops:
                # 滚动均值
                df[f"MA_{window}"] = df["$close"].rolling(window).mean()

            if "STD" in include_ops and "STD" not in exclude_ops:
                # 滚动标准差
                df[f"STD_{window}"] = df["$close"].rolling(window).std()

            if "ROC" in include_ops and "ROC" not in exclude_ops:
                # 滚动变化率（Rate of Change）
                df[f"ROC_{window}"] = df["$close"].pct_change(window)

        # ============================
        # 使用 ta-lib 计算指标
        # ============================
        df = self._compute_talib_features(df)

        # ============================
        # 缺失值处理
        # ============================
        df.fillna(0, inplace=True)

        return df

    def load(self, instruments, start_time, end_time, fields):
        """
        加载数据并应用特征处理。
        """
        # 加载基础数据
        raw_data = super().load(instruments, start_time, end_time, fields)

        # 添加特征计算逻辑
        processed_data = self._process_data(raw_data)
        return processed_data
