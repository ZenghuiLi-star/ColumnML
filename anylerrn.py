import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf

def loadDataSet(fileName):
    xArr = []
    yArr = []
    try:
        fr = pd.read_csv(fileName)
    except:
        fr = pd.read_excel(fileName)

    fr = fr.dropna()

    for i in range(len(fr)):
        line = fr.iloc[i].values
        lineArr = [float(line[i]) for i in range(1, len(line)-1)]
        target = float(line[-1])

        xArr.append(lineArr)
        yArr.append(target)
    return np.array(xArr), np.array(yArr)

def evaluate_regress(y_pre, y_true):

    MAE=np.sum(np.abs(y_pre-y_true))/len(y_true)

    nonzero_mask = y_true != 0
    y_true_nonzero = y_true[nonzero_mask]
    y_pre_nonzero = y_pre[nonzero_mask]

    MAPE = np.sum(np.abs((y_pre_nonzero - y_true_nonzero) / y_true_nonzero)) / len(y_true_nonzero)

    MSE=np.sum((y_pre-y_true) ** 2)/len(y_true)

    RMSE=np.sqrt(MSE)

    R2=r2_score(y_pre, y_true)

    return MAE,MAPE,MSE,RMSE,R2

xArr, yArr = loadDataSet('MY.csv')

data_train_x, data_test_x, data_train_y, data_test_y = train_test_split(xArr, yArr, test_size=0.2, shuffle=False)

X_mean, y_mean = data_train_x.mean(0), data_train_y.mean(0)
X_std, y_std = data_train_x.std(0), data_train_y.std(0)

data_train_x_nor = (data_train_x - X_mean) / X_std
data_test_x_nor = (data_test_x - X_mean) / X_std

data_train_y_nor = (data_train_y - y_mean) / y_std

model_rf = RandomForestRegressor(n_estimators=800, max_depth=9, min_samples_leaf=3, min_samples_split=6, max_features='sqrt', n_jobs=-1)

model_rf.fit(data_train_x, data_train_y)

y_pred_test = model_rf.predict(data_test_x)
y_pred_train = model_rf.predict(data_train_x)

y_pred_test1 = y_pred_test.reshape(len(y_pred_test), 1)
data_test_y = data_test_y.reshape(len(data_test_y), 1)

y_pred_train1 = y_pred_train.reshape(len(y_pred_train), 1)
data_train_y = data_train_y.reshape(len(data_train_y), 1)

data_train_y_df = pd.DataFrame(data_train_y, columns=['y_train_true'])
data_test_y_df = pd.DataFrame(data_test_y, columns=['y_test_true'])

y_train_pred_df = pd.DataFrame(y_pred_train1, columns=['y_train_pred'])
y_test_pred_df = pd.DataFrame(y_pred_test1, columns=['y_test_pred'])

MAE_test, MAPE_test, MSE_test, RMSE_test, R2_test = evaluate_regress(data_test_y, y_pred_test1)
results_test_dict = {
    'MAE': [MAE_test],
    'MAPE': [MAPE_test],
    'MSE': [MSE_test],
    'RMSE': [RMSE_test],
    'R2': [R2_test]
}
results_df_test = pd.DataFrame(results_test_dict)

MAE_train, MAPE_train, MSE_train, RMSE_train, R2_train = evaluate_regress(data_train_y, y_pred_train1)
results_train_dict = {
    'MAE': [MAE_train],
    'MAPE': [MAPE_train],
    'MSE': [MSE_train],
    'RMSE': [RMSE_train],
    'R2': [R2_train]
}
results_df_train = pd.DataFrame(results_train_dict)

rf_data = pd.concat([data_train_y_df, y_train_pred_df, results_df_train, data_test_y_df, y_test_pred_df, results_df_test], axis=1)

rf_data.to_csv('rf_predictions.csv', index=False)

xgb_regressor = xgb.XGBRegressor(n_estimators=500,max_depth=10,learning_rate=0.2,reg_lambda=5,n_jobs=-1)
xgb_regressor.fit(data_train_x, data_train_y)

y_pred_test = xgb_regressor.predict(data_test_x)
y_pred_train = xgb_regressor.predict(data_train_x)

y_pred_test1 = y_pred_test.reshape(len(y_pred_test), 1)
data_test_y = data_test_y.reshape(len(data_test_y), 1)

y_pred_train1 = y_pred_train.reshape(len(y_pred_train), 1)
data_train_y = data_train_y.reshape(len(data_train_y), 1)

data_train_y_df = pd.DataFrame(data_train_y, columns=['y_train_true'])
data_test_y_df = pd.DataFrame(data_test_y, columns=['y_test_true'])

y_train_pred_df = pd.DataFrame(y_pred_train1, columns=['y_train_pred'])
y_test_pred_df = pd.DataFrame(y_pred_test1, columns=['y_test_pred'])

MAE_test, MAPE_test, MSE_test, RMSE_test, R2_test = evaluate_regress(data_test_y, y_pred_test1)
results_test_dict = {
    'MAE': [MAE_test],
    'MAPE': [MAPE_test],
    'MSE': [MSE_test],
    'RMSE': [RMSE_test],
    'R2': [R2_test]
}
results_df_test = pd.DataFrame(results_test_dict)

MAE_train, MAPE_train, MSE_train, RMSE_train, R2_train = evaluate_regress(data_train_y, y_pred_train1)
results_train_dict = {
    'MAE': [MAE_train],
    'MAPE': [MAPE_train],
    'MSE': [MSE_train],
    'RMSE': [RMSE_train],
    'R2': [R2_train]
}
results_df_train = pd.DataFrame(results_train_dict)

data = pd.concat([data_train_y_df, y_train_pred_df, results_df_train, data_test_y_df, y_test_pred_df, results_df_test], axis=1)

data.to_csv('xg_predictions.csv', index=False)

svr_model = SVR(kernel='rbf', C=30, epsilon=0.1, gamma=1)
svr_model.fit(data_train_x_nor, data_train_y_nor)

y_pred_test_nor = svr_model.predict(data_test_x_nor)
y_pred_train_nor = svr_model.predict(data_train_x_nor)

y_pred_test = y_pred_test_nor * y_std + y_mean
y_pred_test1 = y_pred_test.reshape(len(y_pred_test), 1)
data_test_y = data_test_y.reshape(len(data_test_y), 1)

y_pred_train = y_pred_train_nor * y_std + y_mean
y_pred_train1 = y_pred_train.reshape(len(y_pred_train), 1)
data_train_y = data_train_y.reshape(len(data_train_y), 1)

data_train_y_df = pd.DataFrame(data_train_y, columns=['y_train_true'])
data_test_y_df = pd.DataFrame(data_test_y, columns=['y_test_true'])

y_train_pred_df = pd.DataFrame(y_pred_train1, columns=['y_train_pred'])
y_test_pred_df = pd.DataFrame(y_pred_test1, columns=['y_test_pred'])

MAE_test, MAPE_test, MSE_test, RMSE_test, R2_test = evaluate_regress(data_test_y, y_pred_test1)
results_test_dict = {
    'MAE': [MAE_test],
    'MAPE': [MAPE_test],
    'MSE': [MSE_test],
    'RMSE': [RMSE_test],
    'R2': [R2_test]
}
results_df_test = pd.DataFrame(results_test_dict)

MAE_train, MAPE_train, MSE_train, RMSE_train, R2_train = evaluate_regress(data_train_y, y_pred_train1)
results_train_dict = {
    'MAE': [MAE_train],
    'MAPE': [MAPE_train],
    'MSE': [MSE_train],
    'RMSE': [RMSE_train],
    'R2': [R2_train]
}
results_df_train = pd.DataFrame(results_train_dict)

data = pd.concat([data_train_y_df, y_train_pred_df, results_df_train, data_test_y_df, y_test_pred_df, results_df_test], axis=1)

data.to_csv('svm_predictions.csv', index=False)

knn = KNeighborsRegressor(n_neighbors=3, p=1,weights='uniform', n_jobs=-1)  # 使用5个邻居
knn.fit(data_train_x_nor, data_train_y_nor)

y_pred_test_nor = knn.predict(data_test_x_nor)
y_pred_train_nor = knn.predict(data_train_x_nor)

y_pred_test = y_pred_test_nor * y_std + y_mean
y_pred_test1 = y_pred_test.reshape(len(y_pred_test), 1)
data_test_y = data_test_y.reshape(len(data_test_y), 1)

y_pred_train = y_pred_train_nor * y_std + y_mean
y_pred_train1 = y_pred_train.reshape(len(y_pred_train), 1)
data_train_y = data_train_y.reshape(len(data_train_y), 1)

data_train_y_df = pd.DataFrame(data_train_y, columns=['y_train_true'])
data_test_y_df = pd.DataFrame(data_test_y, columns=['y_test_true'])

y_train_pred_df = pd.DataFrame(y_pred_train1, columns=['y_train_pred'])
y_test_pred_df = pd.DataFrame(y_pred_test1, columns=['y_test_pred'])

MAE_test, MAPE_test, MSE_test, RMSE_test, R2_test = evaluate_regress(data_test_y, y_pred_test1)
results_test_dict = {
    'MAE': [MAE_test],
    'MAPE': [MAPE_test],
    'MSE': [MSE_test],
    'RMSE': [RMSE_test],
    'R2': [R2_test]
}
results_df_test = pd.DataFrame(results_test_dict)

MAE_train, MAPE_train, MSE_train, RMSE_train, R2_train = evaluate_regress(data_train_y, y_pred_train1)
results_train_dict = {
    'MAE': [MAE_train],
    'MAPE': [MAPE_train],
    'MSE': [MSE_train],
    'RMSE': [RMSE_train],
    'R2': [R2_train]
}
results_df_train = pd.DataFrame(results_train_dict)

data = pd.concat([data_train_y_df, y_train_pred_df, results_df_train, data_test_y_df, y_test_pred_df, results_df_test], axis=1)

data.to_csv('KNN_predictions.csv', index=False)

tf.config.threading.set_intra_op_parallelism_threads(64)  # 设置内部线程数
tf.config.threading.set_inter_op_parallelism_threads(32)  # 设置不同操作间的线程数

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=data_train_x.shape[1]),  # 第一层
    tf.keras.layers.Dense(64, activation='relu'),  # 第四层
    tf.keras.layers.Dense(32, activation='relu'),  # 第五层
    tf.keras.layers.Dense(1)  # 输出层
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

model.fit(data_train_x_nor, data_train_y_nor, epochs=100, batch_size=32, verbose=0)

y_pred_test_nor = model.predict(data_test_x_nor, verbose=0)
y_pred_train_nor = model.predict(data_train_x_nor, verbose=0)

y_pred_test = y_pred_test_nor * y_std + y_mean
y_pred_test1 = y_pred_test.reshape(len(y_pred_test), 1)
data_test_y = data_test_y.reshape(len(data_test_y), 1)

y_pred_train = y_pred_train_nor * y_std + y_mean
y_pred_train1 = y_pred_train.reshape(len(y_pred_train), 1)
data_train_y = data_train_y.reshape(len(data_train_y), 1)

data_train_y_df = pd.DataFrame(data_train_y, columns=['y_train_true'])
data_test_y_df = pd.DataFrame(data_test_y, columns=['y_test_true'])

y_train_pred_df = pd.DataFrame(y_pred_train1, columns=['y_train_pred'])
y_test_pred_df = pd.DataFrame(y_pred_test1, columns=['y_test_pred'])

MAE_test, MAPE_test, MSE_test, RMSE_test, R2_test = evaluate_regress(data_test_y, y_pred_test1)
results_test_dict = {
    'MAE': [MAE_test],
    'MAPE': [MAPE_test],
    'MSE': [MSE_test],
    'RMSE': [RMSE_test],
    'R2': [R2_test]
}
results_df_test = pd.DataFrame(results_test_dict)

MAE_train, MAPE_train, MSE_train, RMSE_train, R2_train = evaluate_regress(data_train_y, y_pred_train1)
results_train_dict = {
    'MAE': [MAE_train],
    'MAPE': [MAPE_train],
    'MSE': [MSE_train],
    'RMSE': [RMSE_train],
    'R2': [R2_train]
}
results_df_train = pd.DataFrame(results_train_dict)

data = pd.concat([data_train_y_df, y_train_pred_df, results_df_train, data_test_y_df, y_test_pred_df, results_df_test], axis=1)

data.to_csv('ANN_predictions.csv', index=False)

dt_regressor = DecisionTreeRegressor(max_depth=11, min_samples_split=4, min_samples_leaf=3)

dt_regressor.fit(data_train_x, data_train_y)

y_pred_test = dt_regressor.predict(data_test_x)
y_pred_train = dt_regressor.predict(data_train_x)

y_pred_test1 = y_pred_test.reshape(len(y_pred_test), 1)
data_test_y = data_test_y.reshape(len(data_test_y), 1)

y_pred_train1 = y_pred_train.reshape(len(y_pred_train), 1)
data_train_y = data_train_y.reshape(len(data_train_y), 1)

data_train_y_df = pd.DataFrame(data_train_y, columns=['y_train_true'])
data_test_y_df = pd.DataFrame(data_test_y, columns=['y_test_true'])

y_train_pred_df = pd.DataFrame(y_pred_train1, columns=['y_train_pred'])
y_test_pred_df = pd.DataFrame(y_pred_test1, columns=['y_test_pred'])

MAE_test, MAPE_test, MSE_test, RMSE_test, R2_test = evaluate_regress(data_test_y, y_pred_test1)
results_test_dict = {
    'MAE': [MAE_test],
    'MAPE': [MAPE_test],
    'MSE': [MSE_test],
    'RMSE': [RMSE_test],
    'R2': [R2_test]
}
results_df_test = pd.DataFrame(results_test_dict)

MAE_train, MAPE_train, MSE_train, RMSE_train, R2_train = evaluate_regress(data_train_y, y_pred_train1)
results_train_dict = {
    'MAE': [MAE_train],
    'MAPE': [MAPE_train],
    'MSE': [MSE_train],
    'RMSE': [RMSE_train],
    'R2': [R2_train]
}
results_df_train = pd.DataFrame(results_train_dict)

data = pd.concat([data_train_y_df, y_train_pred_df, results_df_train, data_test_y_df, y_test_pred_df, results_df_test], axis=1)

data.to_csv('dt_predictions.csv', index=False)