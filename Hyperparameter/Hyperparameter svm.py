# =========================
# 多文件支持向量机回归模型
# =========================

import numpy as np

from sklearn.svm import SVR
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

    svm_model = SVR()

    param_distributions = [
        {
            'kernel': [
                'rbf'
            ],
            'C': [
                0.01, 0.1, 1,
                10, 50, 100,
                500, 1000
            ],
            'epsilon': [
                0.001, 0.01,
                0.05, 0.1,
                0.2, 0.5, 1.0
            ],
            'gamma': [
                'scale',
                'auto',
                0.0001,
                0.001,
                0.01,
                0.1,
                1.0
            ]
        },
        {
            'kernel': [
                'linear'
            ],
            'C': [
                0.01, 0.1, 1,
                10, 50, 100,
                500, 1000
            ],
            'epsilon': [
                0.001, 0.01,
                0.05, 0.1,
                0.2, 0.5, 1.0
            ]
        },
        {
            'kernel': [
                'poly'
            ],
            'C': [
                0.01, 0.1, 1,
                10, 50, 100,
                500
            ],
            'epsilon': [
                0.001, 0.01,
                0.05, 0.1,
                0.2, 0.5
            ],
            'gamma': [
                'scale',
                'auto',
                0.001,
                0.01,
                0.1,
                1.0
            ],
            'degree': [
                2,
                3,
                4
            ],
            'coef0': [
                0,
                0.1,
                0.5,
                1.0
            ]
        },
        {
            'kernel': [
                'sigmoid'
            ],
            'C': [
                0.01, 0.1, 1,
                10, 50, 100
            ],
            'epsilon': [
                0.001, 0.01,
                0.05, 0.1,
                0.2, 0.5
            ],
            'gamma': [
                'scale',
                'auto',
                0.001,
                0.01,
                0.1
            ],
            'coef0': [
                0,
                0.1,
                0.5,
                1.0
            ]
        }
    ]

    svm_search = RandomizedSearchCV(
        estimator=svm_model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )

    svm_search.fit(
        X_train,
        y_train
    )

    best_svm = svm_search.best_estimator_

    y_pred = best_svm.predict(
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
        -svm_search.best_score_
    )

    print('\n========== 支持向量机结果 ==========')

    print('文件名称：', file_name)

    print('最优参数：')
    print(svm_search.best_params_)

    print(
        '交叉验证最优RMSE：',
        cv_rmse
    )

    print('测试集RMSE：', rmse)
    print('测试集MAE：', mae)
    print('测试集R²：', r2)

    result = {
        '文件名称': file_name,
        'kernel': svm_search.best_params_[
            'kernel'
        ],
        'C': svm_search.best_params_[
            'C'
        ],
        'epsilon': svm_search.best_params_[
            'epsilon'
        ],
        'gamma': svm_search.best_params_.get(
            'gamma',
            None
        ),
        'degree': svm_search.best_params_.get(
            'degree',
            None
        ),
        'coef0': svm_search.best_params_.get(
            'coef0',
            None
        ),
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
    '支持向量机最优超参数汇总.xlsx',
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
    '支持向量机最优超参数汇总.xlsx'
)