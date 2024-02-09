import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



def load_data(path: str, subject: str, labels:list, method="drop"):
    """
    Args:
        path (str): data files directory
        subject (str): subject folder name
        labels (_list_): beta file class
        method (str, optional): Choose to replace NaN values with 'zero' or 'drop' them. Defaults to "drop".

    Returns:
        list, list: a 2D array that has one row of flattened data for each stimuli, corresponding label to each stimul.
    """    
    data = []
    file_paths = [("beta_" + str(x).zfill(4) + ".nii") for x in range(1, 161)]

    for file_path in file_paths:
        arr_cleaned = []
        img = nib.load(path + "BetaPerTrial_betas_" + subject + "/" + file_path)
        img_data = img.get_fdata()
        flattened = img_data.flatten()
        if method == "drop":
            arr_cleaned = flattened[~np.isnan(flattened)]
            print(arr_cleaned)
        elif method == "zero":
            flattened[np.isnan(flattened)] = 0
            arr_cleaned = flattened

        data.append(arr_cleaned)

    return np.array(data), np.array(labels)

