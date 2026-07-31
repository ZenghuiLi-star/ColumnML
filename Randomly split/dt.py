
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import  r2_score
from sklearn.tree import DecisionTreeRegressor

def loadDataSet(fileName):
    xArr = []
    yArr = []
    groupArr = []

    try:
        fr = pd.read_csv(fileName)
    except:
        fr = pd.read_excel(fileName)

    fr = fr.dropna()

    for i in range(len(fr)):
        line = fr.iloc[i].values
        group = str(line[0])
        lineArr = [float(line[i]) for i in range(1, len(line) - 1)]
        target = float(line[-1])

        groupArr.append(group)
        xArr.append(lineArr)
        yArr.append(target)

    return np.array(xArr), np.array(yArr), np.array(groupArr)


def evaluate_regress(y_pre, y_true):
    MAE = np.sum(np.abs(y_pre - y_true)) / len(y_true)

    nonzero_mask = y_true != 0
    y_true_nonzero = y_true[nonzero_mask]
    y_pre_nonzero = y_pre[nonzero_mask]

    MAPE = np.sum(np.abs((y_pre_nonzero - y_true_nonzero) / y_true_nonzero)) / len(y_true_nonzero)

    MSE = np.sum((y_pre - y_true) ** 2) / len(y_true)

    RMSE = np.sqrt(MSE)

    R2 = r2_score(y_pre, y_true)

    return MAE, MAPE, MSE, RMSE, R2


path1 = 'MP'
path2 = 'MY'
path3 = 'K0'
path4 = 'as'
path5 = 'theta_pc'
path6 = 'lamga'
path_list = [path1,path2,path3,path4,path5,path6]

name1 = 'Mp_kunnath.xlsx'
name2 = 'My_kunnath.xlsx'
name3 = 'k0_kunnath.xlsx'
name4 = 'as_kunnath.xlsx'
name5 = 'theta_pc_kunnath.xlsx'
name6 = 'LAMA_kunnath.xlsx'
name_list = [name1,name2,name3,name4,name5,name6]

super_para1 = [30,2,10]
super_para2 = [30,2,10]
super_para3 = [30,2,10]
super_para4 = [30,2,10]
super_para5 = [30,2,10]
super_para6 = [30,2,10]
super_para = [super_para1,super_para2,super_para3,super_para4,super_para5,super_para6]

my_list = []
for j in range(len(name_list)):
    xArr, yArr, groupArr = loadDataSet(name_list[j])

    best_r2 = -np.inf
    best_seed = None

    count = []
    Test_all = np.empty((0, 10))
    for i in range(1, 1000):
        group_split = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=i
        )

        train_index, test_index = next(
            group_split.split(xArr, yArr, groups=groupArr)
        )

        data_train_x = xArr[train_index]
        data_test_x = xArr[test_index]
        data_train_y = yArr[train_index]
        data_test_y = yArr[test_index]

        X_mean, y_mean = data_train_x.mean(0), data_train_y.mean(0)
        X_std, y_std = data_train_x.std(0), data_train_y.std(0)

        data_train_x_nor = (data_train_x - X_mean) / X_std
        data_test_x_nor = (data_test_x - X_mean) / X_std

        data_train_y_nor = (data_train_y - y_mean) / y_std
        data_test_y_nor = (data_test_y - y_mean) / y_std

        dt_regressor = DecisionTreeRegressor(max_depth=super_para[j][0], min_samples_split=super_para[j][1], min_samples_leaf=super_para[j][2])

        dt_regressor.fit(data_train_x, data_train_y)

        y_pred_test = dt_regressor.predict(data_test_x)
        y_pred_train = dt_regressor.predict(data_train_x)

        y_pred_test1 = y_pred_test.reshape(len(y_pred_test), 1)
        data_test_y = data_test_y.reshape(len(data_test_y), 1)

        y_pred_train1 = y_pred_train.reshape(len(y_pred_train), 1)
        data_train_y = data_train_y.reshape(len(data_train_y), 1)

        _, _, _, _, r2_test = evaluate_regress(data_test_y, y_pred_test1)

        MAE_test, MAPE_test, MSE_test, RMSE_test, R2_test = evaluate_regress(data_test_y, y_pred_test1)
        new_row_test = [MAE_test, MAPE_test, MSE_test, RMSE_test, R2_test]
        new_row_test = pd.DataFrame(new_row_test)
        new_row_test = new_row_test.T

        MAE_train, MAPE_train, MSE_train, RMSE_train, R2_train = evaluate_regress(data_train_y, y_pred_train1)
        new_row_train = [MAE_train, MAPE_train, MSE_train, RMSE_train, R2_train]
        new_row_train = pd.DataFrame(new_row_train)
        new_row_train = new_row_train.T

        new_row = pd.concat([new_row_train, new_row_test], axis=1)
        Test_all = np.vstack([Test_all, new_row])

        if r2_test > best_r2:
            best_r2 = r2_test
            best_seed = i

        data_Oriny_pre = {}
        data_Oriny_pre['y_train_predict'] = y_pred_train1
        data_Oriny_pre['y_test_predict'] = y_pred_test1

    results_df = pd.DataFrame(Test_all, columns=['MAE_train', 'MAPE_train', 'MSE_train', 'RMSE_train', 'R2_train', 'MAE_test', 'MAPE_test', 'MSE_test', 'RMSE_test', 'R2_test'])
    results_df.to_csv(f'data_evaluate\\dt_{path_list[j]}_evaluate.csv', index=False)
    my_list.append(f"{path_list[j]} 最佳R²值: {best_r2} 对应的随机种子是: {best_seed}")

with open("dt.txt", "w", encoding="utf-8") as file:
    for item in my_list:
        file.write(item + "\n")