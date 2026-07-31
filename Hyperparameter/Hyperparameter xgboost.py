# =========================
# 多文件XGBoost回归模型
# =========================

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        data_train_x,
        dtype=float
    )

    y_train = np.asarray(
        data_train_y,
        dtype=float
    ).ravel()

    X_test = np.asarray(
        data_test_x,
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

    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        random_state=42,
        n_jobs=1,
        verbosity=0
    )

    param_distributions = {
        'n_estimators': [
            100, 200, 300,
            500, 800, 1000
        ],

        'max_depth': [
            2, 3, 4, 5,
            6, 8, 10
        ],

        'learning_rate': [
            0.01, 0.03, 0.05,
            0.1, 0.15, 0.2
        ],

        'min_child_weight': [
            1, 2, 3,
            5, 7, 10
        ],

        'subsample': [
            0.6, 0.7, 0.8,
            0.9, 1.0
        ],

        'colsample_bytree': [
            0.6, 0.7, 0.8,
            0.9, 1.0
        ],

        'gamma': [
            0, 0.01, 0.05,
            0.1, 0.2, 0.5
        ],

        'reg_alpha': [
            0, 0.001,
            0.01, 0.1, 1.0
        ],

        'reg_lambda': [
            0.1, 0.5,
            1.0, 5.0, 10.0
        ]
    }

    xgb_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )

    xgb_search.fit(
        X_train,
        y_train
    )

    best_xgb = xgb_search.best_estimator_

    y_pred = best_xgb.predict(
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
        -xgb_search.best_score_
    )

    print('\n========== XGBoost结果 ==========')

    print('文件名称：', file_name)

    print('最优参数：')
    print(xgb_search.best_params_)

    print(
        '交叉验证最优RMSE：',
        cv_rmse
    )

    print('测试集RMSE：', rmse)
    print('测试集MAE：', mae)
    print('测试集R²：', r2)

    result = {
        '文件名称': file_name,
        'n_estimators': xgb_search.best_params_[
            'n_estimators'
        ],
        'max_depth': xgb_search.best_params_[
            'max_depth'
        ],
        'learning_rate': xgb_search.best_params_[
            'learning_rate'
        ],
        'min_child_weight': xgb_search.best_params_[
            'min_child_weight'
        ],
        'subsample': xgb_search.best_params_[
            'subsample'
        ],
        'colsample_bytree': xgb_search.best_params_[
            'colsample_bytree'
        ],
        'gamma': xgb_search.best_params_[
            'gamma'
        ],
        'reg_alpha': xgb_search.best_params_[
            'reg_alpha'
        ],
        'reg_lambda': xgb_search.best_params_[
            'reg_lambda'
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
    'XGBoost最优超参数汇总.xlsx',
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
    'XGBoost最优超参数汇总.xlsx'
)