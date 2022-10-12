#%%
from kdg import kdf
from kdg.utils import get_ece
import openml
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import cohen_kappa_score
from kdg.utils import get_ece
import os
from os import listdir, getcwd 
# %%
def count_rf_param(rf_model):
    total_param = 0
    for tree in rf_model.estimators_:
        nodes = tree.tree_.node_count
        leaf_node = np.sum(tree.tree_.children_left)
        total_param += tree.tree_.value.shape[2]*leaf_node\
            + (nodes-leaf_node)*2 + nodes
    return total_param

def count_kdf_param(kdf_model):
    total_param = 0

    for label in kdf_model.labels:
        total_param += len(kdf_model.polytope_cardinality[label])
    
    total_param += len(kdf_model.polytope_cardinality[label])\
            *(kdf_model.feature_dim*2)

    return total_param

def experiment_random_sample(dataset_id, folder, n_estimators=500, reps=10):
    
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, is_categorical, _ = dataset.get_data(
                dataset_format="array", target=dataset.default_target_attribute
            )

    if np.mean(is_categorical) >0:
        return

    if np.isnan(np.sum(y)):
        return

    if np.isnan(np.sum(X)):
        return
    
    print("doing ",dataset_id) 

    total_sample = X.shape[0]

    test_sample = total_sample//3

    max_sample = total_sample - test_sample
    train_samples = np.logspace(
        np.log10(2),
        np.log10(max_sample),
        num=10,
        endpoint=True,
        dtype=int
        )
    
    err = []
    err_rf = []
    ece = []
    ece_rf = []
    kappa = []
    kappa_rf = []
    mc_rep = []
    samples = []
    param_kdf = []
    param_rf = []
    indices = list(range(total_sample))

    for train_sample in train_samples:     
        for rep in range(reps):
            np.random.shuffle(indices)
            indx_to_take_train = indices[:train_sample]
            indx_to_take_test = indices[-test_sample:]

            model_kdf = kdf(kwargs={'n_estimators':n_estimators})
            model_kdf.fit(X[indx_to_take_train], y[indx_to_take_train], epsilon=1e-6)
            proba_kdf = model_kdf.predict_proba(X[indx_to_take_test])
            proba_rf = model_kdf.rf_model.predict_proba((X[indx_to_take_test]-model_kdf.min_val)/(model_kdf.max_val-model_kdf.min_val+1e-8))
            predicted_label_kdf = np.argmax(proba_kdf, axis = 1)
            predicted_label_rf = np.argmax(proba_rf, axis = 1)

            err.append(
                1 - np.mean(
                        predicted_label_kdf==y[indx_to_take_test]
                    )
            )
            err_rf.append(
                1 - np.mean(
                    predicted_label_rf==y[indx_to_take_test]
                )
            )
            kappa.append(
                cohen_kappa_score(predicted_label_kdf, y[indx_to_take_test])
            )
            kappa_rf.append(
                cohen_kappa_score(predicted_label_rf, y[indx_to_take_test])
            )
            ece.append(
                get_ece(proba_kdf, predicted_label_kdf, y[indx_to_take_test])
            )
            ece_rf.append(
                get_ece(proba_rf, predicted_label_rf, y[indx_to_take_test])
            )
            samples.append(
                train_sample
            )
            param_kdf.append(
                count_kdf_param(model_kdf)
            )
            param_rf.append(
                count_rf_param(model_kdf.rf_model)
            )
            mc_rep.append(rep)

    df = pd.DataFrame() 
    df['err_kdf'] = err
    df['err_rf'] = err_rf
    df['kappa_kdf'] = kappa
    df['kappa_rf'] = kappa_rf
    df['ece_kdf'] = ece
    df['ece_rf'] = ece_rf
    df['rep'] = mc_rep
    df['samples'] = samples
    df['kdf_param'] = param_kdf
    df['rf_param'] = param_rf

    df.to_csv(folder+'/'+'openML_cc18_'+str(dataset_id)+'.csv')


#%%
folder = 'openml_res'
#os.mkdir(folder)
#os.mkdir(folder_rf)
benchmark_suite = openml.study.get_suite('OpenML-CC18')
#current_dir = getcwd()
#files = listdir(current_dir+'/'+folder)
Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment_random_sample)(
                dataset_id,
                folder
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )
'''for dataset_id in openml.study.get_suite("OpenML-CC18").data:
    print('doing ',dataset_id)
    experiment_random_sample(
                    dataset_id,
                    folder
                    ) '''

'''Parallel(n_jobs=-1,verbose=1)(
        delayed(experiment_rf)(
                dataset_id,
                folder_rf
                ) for dataset_id in openml.study.get_suite("OpenML-CC18").data
            )'''
'''for task_id in benchmark_suite.tasks:
    filename = 'openML_cc18_' + str(task_id) + '.csv'

    if filename not in files:
        print(filename)
        try:
            experiment(task_id,folder)
        except:
            print("couldn't run!")
        else:
            print("Ran successfully!")'''
# %%
