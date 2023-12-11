# -*- coding: utf-8 -*-
from functools import reduce

import torch
from sklearn.preprocessing import minmax_scale, scale

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, average_precision_score, f1_score
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor as Executor

import numpy as np
import argparse
import numpy as np
import  pandas as pd
import argparse
from sko.PSO import PSO
from sko.operators import ranking, selection, crossover, mutation


np.random.seed(22)

def set_args():
    parser = argparse.ArgumentParser(description='Train SVM') 

    parser.add_argument('--x_path', type=str, default=None,
                        help='The path to feature files.', nargs='+')
    parser.add_argument('--y_path', type=str, default=None,
                        help='The path to label files.')

    args = parser.parse_args()

    return args


def features_labels_load(feature_path, y_path):

    features =[]
    symbol_all =[]
    for i in feature_path:
        # 将gcn中0值过多的值清洗
        if i.split('.')[0][-3:] == 'gcn':
            fear = np.load(i, allow_pickle=True)
            temp_vec_df = pd.DataFrame(fear['features'], index=fear['symbol'])
            temp_vec_df = temp_vec_df.loc[~(temp_vec_df == 0).all(axis=1)]
            features.append(np.array(temp_vec_df))
            symbol_all.append(np.array(temp_vec_df.index))
        else:
            features.append(np.load(i, allow_pickle=True)["features"] )
            symbol_all.append(np.load(i, allow_pickle=True)["symbol"])




    if len(symbol_all) > 1:
        symbol = list(set(reduce(lambda x,y: list(x)+list(y), list(symbol_all))))
        symbol.sort()

        dict_symbol = dict(zip(symbol, range(len(symbol))))
        
        features_union = np.zeros((len(symbol_all), len(symbol), features[0].shape[1]))
        for i in range(len(symbol_all)):
            mask = np.array([dict_symbol[j] for j in symbol_all[i]])
            features_union[i][mask] = features[i]
            
        features = np.concatenate(features_union, axis=1)
    else:
        symbol = symbol_all[0]
        features = features[0]

    postive_symbol = np.load(y_path)
    label = np.array([i in postive_symbol for i in symbol])
    
    pos_index, neg_index = [i for i, j in enumerate(label) if j == 1], [i for i, j in enumerate(label) if j == 0]
    neg_index_choice = np.random.choice(neg_index, len(pos_index), replace=False)
    all_index = np.concatenate([pos_index, neg_index_choice], axis=0)
    features = features[all_index]
    label = label[all_index]

    # print(label.shape)

    return features, label

def worker(x_train, x_valid, y_train, y_valid, C=1.0):
# def worker(x_train, x_valid, y_train, y_valid,
#            mustlinks):
    model = SVC(C=C, kernel='precomputed', class_weight='balanced', random_state=10, probability=True, verbose=False)

    model.fit(x_train, y_train)

    y_valid_score = model.predict_proba(x_valid)[:, 1]

    aupr, auc = average_precision_score(y_valid, y_valid_score), roc_auc_score(y_valid, y_valid_score)
    acc, f1_ =  accuracy_score(y_valid, y_valid_score>0.5), f1_score(y_valid, y_valid_score>0.5)
    

    return auc, aupr, acc, f1_


def run(X, y, gamma, C):
    X = rbf_kernel(X, gamma=gamma)

    kfold = StratifiedKFold(n_splits=10, random_state=22, shuffle=True)

    tasks, results, true_pred = [], [], []
    # Executor是一个抽象类，它提供了异步执行调用的方法。它不能直接使用，但可以通过它的两个子类ThreadPoolExecutor或者ProcessPoolExecutor进行调用
    with Executor(max_workers=10) as executor:

        for train_index, valid_index in kfold.split(X, y):
            x_train, x_valid = X[train_index][:, train_index], X[valid_index][:, train_index]
            y_train, y_valid = y[train_index], y[valid_index]
            
            tasks.append(executor.submit(worker, x_train, x_valid, y_train, y_valid, C))

        for future in as_completed(tasks):
            results.append(list(future.result()))

    # print("#########  gamma: {}, C: {} ###########".format(gamma, C))
    # print(np.mean(np.array(results), axis=0))

    return np.mean(np.array(results), axis=0)

def main():
    args = set_args()
    x_path, y_path = args.x_path, args.y_path
    X, y = features_labels_load(x_path, y_path)
    res = 0
    all_result = []

    def func(arg):
        gamma, C = arg
        result = run(X, y, gamma, C)
        nonlocal res
        nonlocal all_result
        if res > -result[0]:
            res = -result[0]
            all_result = result
            print(res)
        # f1 = f1_score(y_test, model.predict(x_test), average="macro")
        return res
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pso = PSO(func=func, dim=2, pop=60, max_iter=200, lb=[0.0001, 1], ub=[1, 25], w=0.9, c1=1.6, c2=2)
    # pso = PSO(func=func, dim=2, pop=60, max_iter=300, lb=[0.0001, 1], ub=[1, 25])

    pso.record_mode = True

    pso.run()

    print('best_x:', pso.gbest_x, '\n', 'best_y:', pso.gbest_y)
    # print('每一代的每个个体的适应度:', ga.all_history_FitV, '\n', '每一代每个个体的函数值:',ga.all_history_Y)

    # 输出结果
    best_C, best_gamma = pso.gbest_x[1], pso.gbest_x[0]
    best_result = all_result
    print("### best_gamma: {}, best_C: {} ###".format(best_gamma, best_C))
    # print(best_result)
    print('AUROC: {:.3f}\tAUPRC: {:.3f}\tAcc: {:.3f}\tF1: {:.3f}'.format(best_result[0], best_result[1], best_result[2],
                                                                         best_result[3]))
    # 保存当次结果
    # file_handle = open('res.txt', mode='w')
    # file_handle.writelines([best_result[0]+'\n',best_result[1]+'\n', best_result[2]+'\n', best_result[2]+'\n'])
    #
    # file_handle.close()

'''
def main():

    args = set_args()
    x_path, y_path = args.x_path, args.y_path
    X, y = features_labels_load(x_path, y_path)

    gammas = [0.001, 0.01, 0.1, 1.0]
    Cs = [0.1, 1.0, 10.0, 100.0]
    
    best_auc = 0.
    best_C, best_gamma = 0., 0.
    for gamma in gammas:
        for C in Cs:
            result = run(X, y, gamma, C)
            if result[0] > best_auc:
                best_auc = result[0]

                best_C, best_gamma = C, gamma 
                best_result = result 
    
    print("### best_gamma: {}, best_C: {} ###".format(best_gamma, best_C))
    # print(best_result)
    print('AUROC: {:.3f}\tAUPRC: {:.3f}\tAcc: {:.3f}\tF1: {:.3f}'.format(best_result[0], best_result[1], best_result[2], best_result[3]))
'''
if __name__ == '__main__':
    main()
