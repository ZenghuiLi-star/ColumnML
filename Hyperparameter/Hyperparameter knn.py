# =========================
# 多文件K近邻回归模型
# =========================

import numpy as np

from sklearn.neighbors import KNeighborsRegressor
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

    knn_model = KNeighborsRegressor(
        n_jobs=1
    )

    param_distributions = {
        'n_neighbors': [
            1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            12, 15, 20, 25, 30
        ],
        'weights': [
            'uniform',
            'distance'
        ],
        'algorithm': [
            'auto',
            'ball_tree',
            'kd_tree',
            'brute'
        ],
        'leaf_size': [
            10,
            20,
            30,
            40,
            50,
            60
        ],
        'p': [
            1,
            2
        ]
    }

    knn_search = RandomizedSearchCV(
        estimator=knn_model,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )

    knn_search.fit(
        X_train,
        y_train
    )

    best_knn = knn_search.best_estimator_

    y_pred = best_knn.predict(
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
        -knn_search.best_score_
    )

    print('\n========== K近邻结果 ==========')

    print('文件名称：', file_name)

    print('最优参数：')
    print(knn_search.best_params_)

    print(
        '交叉验证最优RMSE：',
        cv_rmse
    )

    print('测试集RMSE：', rmse)
    print('测试集MAE：', mae)
    print('测试集R²：', r2)

    result = {
        '文件名称': file_name,
        'n_neighbors': knn_search.best_params_[
            'n_neighbors'
        ],
        'weights': knn_search.best_params_[
            'weights'
        ],
        'algorithm': knn_search.best_params_[
            'algorithm'
        ],
        'leaf_size': knn_search.best_params_[
            'leaf_size'
        ],
        'p': knn_search.best_params_[
            'p'
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
    'K近邻最优超参数汇总.xlsx',
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
    'K近邻最优超参数汇总.xlsx'
)