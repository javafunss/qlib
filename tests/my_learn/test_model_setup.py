
"""训练和调优机器学习模型步骤"""

# %%
# 1.加载数据
from qlib.data import FeatureProcessor
from pandas import Timestamp
import qlib
from qlib.data import D
from qlib.data.filter import NameDFilter
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')

# 加载数据
instruments = ["SH600000"]
data = D.features(instruments=instruments, fields=['$close', '$volume'], start_time='2010-01-01', end_time='2020-12-31')

# %%
# 2.数据预处理
# 创建特征处理器
fp = FeatureProcessor()

# 添加技术指标
data = fp.add_technical_indicator(data, ['MA', 'RSI']

# %%
# 3.数据切分
# 为了评估模型的泛化能力，通常需要将数据划分为训练集、验证集和测试集。
from sklearn.model_selection import train_test_split

# 假设 data 是你的特征矩阵，labels 是你的标签
train_data, test_data, train_labels, test_labels=train_test_split(data, labels, test_size=0.2, random_state=42)
train_data, val_data, train_labels, val_labels=train_test_split(train_data, train_labels, test_size=0.25, random_state=42)

# %%
# 4.模型训练
from sklearn.linear_model import LinearRegression  # LightGBM
from lightgbm import LGBMRegressor
import lightgbm as lgb

# 定义参数
params={
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 创建数据集
lgb_train=lgb.Dataset(train_data, train_labels)
lgb_val=lgb.Dataset(val_data, val_labels, reference=lgb_train)

# 训练模型
gbm=lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=10)

# %%
# MLP
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 定义 MLP 模型
    class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1=nn.Linear(input_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, output_dim)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x

    # 初始化模型
    input_dim=train_data.shape[1]
    hidden_dim=64
    output_dim=1
    model=MLP(input_dim, hidden_dim, output_dim)

    # 定义损失函数和优化器
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs=100
    for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs=model(torch.tensor(train_data, dtype=torch.float32))
    loss=criterion(outputs, torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# %%
# LSTM
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 定义 LSTM 模型
    class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc=nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0=torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0=torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _=self.lstm(x, (h0, c0))
        out=self.fc(out[:, -1, :])
        return out

    # 初始化模型
    input_dim=train_data.shape[2]  # 假设 train_data 的形状是 (batch_size, seq_len, input_dim)
    hidden_dim=64
    output_dim=1
    num_layers=2
    model=LSTMModel(input_dim, hidden_dim, output_dim, num_layers)

    # 定义损失函数和优化器
    criterion=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs=100
    for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs=model(torch.tensor(train_data, dtype=torch.float32))
    loss=criterion(outputs, torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# %%
# 5.模型调优
# 模型调优通常包括超参数搜索和交叉验证

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid={
'num_leaves': [31, 63],
'learning_rate': [0.05, 0.1],
'feature_fraction': [0.8, 0.9],
'bagging_fraction': [0.7, 0.8],
'bagging_freq': [5, 10]
}

# 创建 GridSearchCV 对象
grid_search=GridSearchCV(lgb.LGBMRegressor(**params), param_grid, cv=3, scoring='neg_mean_squared_error')

# 执行超参数搜索
grid_search.fit(train_data, train_labels)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)


# %%
# 6.模型评估
# 模型评估包括预测准确率、回归系数、R squared 等指标。交叉验证
from sklearn.model_selection import cross_val_score

# 定义模型
model=lgb.LGBMRegressor(**params)

# 执行交叉验证
scores=cross_val_score(model, train_data, train_labels, cv=5, scoring='neg_mean_squared_error')

# 输出交叉验证结果
print("Cross-validation scores: ", scores)
print("Mean score: ", scores.mean())

# %%
# 模型评估

# 评估 LightGBM 模型
predictions=gbm.predict(test_data)
mse=((predictions - test_labels) ** 2).mean()
print(f'Test MSE: {mse:.4f}')

# 评估 MLP 模型
model.eval()
with torch.no_grad():
predictions=model(torch.tensor(test_data, dtype=torch.float32)).squeeze().numpy()
mse=((predictions - test_labels) ** 2).mean()
print(f'Test MSE: {mse:.4f}')

# 评估 LSTM 模型
model.eval()
with torch.no_grad():
predictions=model(torch.tensor(test_data, dtype=torch.float32)).squeeze().numpy()
mse=((predictions - test_labels) ** 2).mean()
print(f'Test MSE: {mse:.4f}')

# %%
# 回测
from qlib.backtest import BacktestEngine

# 初始化回测引擎
engine=BacktestEngine()

# 设置回测参数
engine.set_backtest_start_date('2020-01-01')
engine.set_backtest_end_date('2020-12-31')
engine.set_benchmark('SH000300')

# 运行回测
engine.run_backtest(predictions)

# 获取回测结果
backtest_results=engine.get_backtest_result()
print(backtest_results)
