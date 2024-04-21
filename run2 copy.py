from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold
import logging
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr


from globals import subjects, txt_labels
from functions import load_data


from collections import Counter


def getEmbeddings(sentences, method = 'bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(method)
    model = AutoModel.from_pretrained(method)
    y = []
    for s in tqdm(sentences):
        tokens = tokenizer(s, return_tensors="pt")
        outputs = model(**tokens)

        em = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        y.append(em)
    return np.array(y)

def corr_score_column(y_test, preds):
    feature_corrs = []
    for j in range(y_test.shape[1] - 1):
        # Calculate column-wise correlation.
        feature_corrs.append(pearsonr(y_test[:,j], preds[:, j])[0])
    
    roe_mean = np.mean(feature_corrs)
    return roe_mean, feature_corrs

def corr_score_row(y_test, preds):
    feature_corrs = []
    for j in range(y_test.shape[0] - 1):
        # Calculate column-wise correlation.
        feature_corrs.append(pearsonr(y_test[j,:], preds[j,:])[0])
    
    roe_mean = np.mean(feature_corrs)
    return roe_mean, feature_corrs


path = "/home/ellie/Desktop/UAlberta/CMPUT605/Code/Gonzo/"
NaN_method = "drop"
classifier="Ridge"
emb_meth='bert'
y_emb = 'yes'

log_path = (
        "/home/ellie/Desktop/UAlberta/CMPUT605/Code/CMPUT605/logs/Decoding/"
        + emb_meth
        + "_"
        + y_emb
        +"_"
        + NaN_method
        + ".log"
    )

logging.basicConfig(filename=log_path, level=logging.INFO,format="%(message)s" )
logging.info(f"Classifier: {classifier}\n NaN Method: {NaN_method}")



sentences = list(map(lambda x : x.replace('_', ' '), pd.read_excel('./stimNamesAndTypeforGONZO.xlsx')['stim'].values) )



corr_all =[]
mse_all=[]

for subj in subjects:
    logging.info(f"\n_______________________Subject: {subj}_______________________")

    beta, _ = load_data(path=path, subject=subj, labels=txt_labels, method=NaN_method)
    scaler = StandardScaler()
    standard_data = scaler.fit_transform(beta)

    desired_variance_ratio = 0.85
    pca = PCA(n_components=desired_variance_ratio)
    X = pca.fit_transform(standard_data)

    embeddings = getEmbeddings(sentences)

    pca = PCA(n_components=20)
    Y = pca.fit_transform(embeddings)
    # Y = embeddings


    # loo = LeaveOneOut()
    loo = KFold(5)
    outer_cv_results = []
    outer_corrs = []
    fold = 0
    for train_index, test_index in tqdm(loo.split(X)):
        fold+=1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
        # Inner cross-validation for hyperparameter tuning
        ridge = Ridge()
        inner_cv = GridSearchCV(estimator=ridge, param_grid=ridge_params, scoring='neg_mean_squared_error', cv=5)
        inner_cv.fit(X_train, y_train)

        best_alpha = inner_cv.best_params_['alpha']

        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_train, y_train)

        # Evaluate on the test set
        y_pred = ridge.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        outer_cv_results.append(mse)
        corr = corr_score_column(y_test, y_pred)[0]
        outer_corrs.append(corr)
        print(corr, mse)
        logging.info(f"\nFold:\n{fold}")
        logging.info(f"\nMSE:\n{mse}")
        logging.info(f"\nCorr:\n{corr}")
    corr_all.append(np.average(outer_corrs))
    mse_all.append(np.average(outer_cv_results))
    logging.info(f"\n-----Subject Results:\n")
    logging.info(f"\nAverage MSE: {np.average(outer_cv_results)}")
    logging.info(f"\nAverage Corr: {np.average(outer_corrs)}")



        



    print('--------------per-----------------')
    observeed_corr = np.mean(outer_corrs)
    num_permutations = 20
    permuted_corrs = []

    for _ in range(num_permutations):
        y_permuted = np.random.permutation(Y)
        loo = KFold(5)
        outer_cv_results = []
        outer_corrs = []


        for train_index, test_index in tqdm(loo.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y_permuted[train_index], y_permuted[test_index]

            ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
            # Inner cross-validation for hyperparameter tuning
            ridge = Ridge()
            inner_cv = GridSearchCV(estimator=ridge, param_grid=ridge_params, scoring='neg_mean_squared_error', cv=5)
            inner_cv.fit(X_train, y_train)

            best_alpha = inner_cv.best_params_['alpha']
            # best_alpha = 1

            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X_train, y_train)

            # Evaluate on the test set
            y_pred = ridge.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            outer_cv_results.append(mse)
            corr = corr_score_column(y_test, y_pred)[0]
            outer_corrs.append(corr)
        permuted_corrs.append(np.mean(outer_corrs))
        
   
    logging.info(f"p-value:{np.sum(np.array(permuted_corrs) >= observeed_corr) / num_permutations}")

logging.info(f"\n_______________________All Subjects Results_______________________\n")
logging.info(f"\nAverage MSE:{np.average(mse_all)}")
logging.info(f"\nAverage Corr:{np.average(corr_all)}")
