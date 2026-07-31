# =========================
# 多文件人工神经网络回归模型
# =========================

import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pandas as pd
from sklearn.model_selection import train_test_split


name1 = 'Mp_kunnath.xlsx'
name2 = 'My_kunnath.xlsx'
name3 = 'k0_kunnath.xlsx'
name4 = 'as_kunnath.xlsx'
name5 = 'theta_pc_kunnath.xlsx'
name6 = 'LAMA_kunnath.xlsx'

file_names = [
    name1,
    name2,
    name3,
    name4,
    name5,
    name6
]

all_results = []


for file_name in file_names:

    print('\n')
    print('=' * 60)
    print('当前处理文件：', file_name)
    print('=' * 60)

    df = pd.read_excel(
        file_name
    ).dropna().reset_index(drop=True)

    split_idx = int(len(df) * 0.25)

    df1 = df.iloc[:split_idx].reset_index(drop=True)
    df2 = df.iloc[split_idx:].reset_index(drop=True)

    df1 = df1.dropna()
    df2 = df2.dropna()

    train1, test1 = train_test_split(
        df1,
        test_size=0.2,
        shuffle=False
    )

    train2, test2 = train_test_split(
        df2,
        test_size=0.2,
        random_state=6
    )

    train_df = pd.concat(
        [train1, train2],
        ignore_index=True
    )

    test_df = pd.concat(
        [test1, test2],
        ignore_index=True
    )

    print(
        'sheet1 训练集:',
        len(train1),
        '测试集:',
        len(test1)
    )

    print(
        'sheet2 训练集:',
        len(train2),
        '测试集:',
        len(test2)
    )

    print(
        '合并后训练集:',
        len(train_df),
        '合并后测试集:',
        len(test_df)
    )

    def to_xy(dataframe):
        x = dataframe.iloc[:, 1:-1].values.astype(float)
        y = dataframe.iloc[:, -1].values.astype(float)
        return x, y

    data_train_x, data_train_y = to_xy(train_df)
    data_test_x, data_test_y = to_xy(test_df)

    X_mean = data_train_x.mean(0)
    y_mean = data_train_y.mean(0)

    X_std = data_train_x.std(0)
    y_std = data_train_y.std(0)

    X_std[X_std == 0] = 1

    data_train_x_nor = (
        data_train_x - X_mean
    ) / X_std

    data_test_x_nor = (
        data_test_x - X_mean
    ) / X_std

    data_train_y_nor = (
        data_train_y - y_mean
    ) / y_std

    data_test_y_nor = (
        data_test_y - y_mean
    ) / y_std

    X_train = np.asarray(
        data_train_x_nor,
        dtype=float
    )

    y_train = np.asarray(
        data_train_y,
        dtype=float
    ).ravel()

    X_test = np.asarray(
        data_test_x_nor,
        dtype=float
    )

    y_test = np.asarray(
        data_test_y,
        dtype=float
    ).ravel()

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    ann_model = MLPRegressor(
        max_iter=5000,
        random_state=42
    )

    param_distributions = {
        'hidden_layer_sizes': [
            (8,),
            (16,),
            (32,),
            (64,),
            (128,),
            (16, 8),
            (32, 16),
            (64, 32),
            (128, 64),
            (32, 16, 8),
            (64, 32, 16),
            (128, 64, 32)
        ],
        'activation': [
            'relu',
            'tanh',
            'logistic'
        ],
        'solver': [
            'adam',
            'lbfgs'
        ],
        'alpha': [
            0.000001,
            0.00001,
            0.0001,
            0.001,
            0.01,
            0.1
        ],
        'learning_rate': [
            'constant',
            'adaptive'
        ],
        'learning_rate_init': [
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05
        ],
        'batch_size': [
            8,
            16,
            32,
            64,
            'auto'
        ],
        'early_stopping': [
            True,
            False
        ],
        'n_iter_no_change': [
            20,
            50,
            100
        ]
    }

    ann_search = RandomizedSearchCV(
        estimator=ann_model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )

    ann_search.fit(
        X_train,
        y_train
    )

    best_ann = ann_search.best_estimator_

    y_pred = best_ann.predict(
        X_test
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    cv_rmse = np.sqrt(
        -ann_search.best_score_
    )

    print('\n========== 人工神经网络结果 ==========')

    print('文件名称：', file_name)

    print('最优参数：')
    print(ann_search.best_params_)

    print(
        '交叉验证最优RMSE：',
        cv_rmse
    )

    print('测试集RMSE：', rmse)
    print('测试集MAE：', mae)
    print('测试集R²：', r2)

    result = {
        '文件名称': file_name,
        'hidden_layer_sizes': str(
            ann_search.best_params_[
                'hidden_layer_sizes'
            ]
        ),
        'activation': ann_search.best_params_[
            'activation'
        ],
        'solver': ann_search.best_params_[
            'solver'
        ],
        'alpha': ann_search.best_params_[
            'alpha'
        ],
        'learning_rate': ann_search.best_params_[
            'learning_rate'
        ],
        'learning_rate_init': ann_search.best_params_[
            'learning_rate_init'
        ],
        'batch_size': ann_search.best_params_[
            'batch_size'
        ],
        'early_stopping': ann_search.best_params_[
            'early_stopping'
        ],
        'n_iter_no_change': ann_search.best_params_[
            'n_iter_no_change'
        ],
        '交叉验证RMSE': cv_rmse,
        '测试集RMSE': rmse,
        '测试集MAE': mae,
        '测试集R2': r2
    }

    all_results.append(
        result
    )


results_df = pd.DataFrame(
    all_results
)

results_df.to_excel(
    '人工神经网络最优超参数汇总.xlsx',
    index=False
)

print('\n')
print('=' * 60)
print('所有文件运行完成')
print('=' * 60)

print(
    results_df.to_string(
        index=False
    )
)

print(
    '\n最优超参数已保存至：'
    '人工神经网络最优超参数汇总.xlsx'
)