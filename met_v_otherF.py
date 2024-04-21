import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import logging
import pandas as pd



from globals import subjects, txt_labels,mVof
from functions import load_data


from collections import Counter


def main():

    path = "/home/ellie/Desktop/UAlberta/CMPUT605/Code/Gonzo/"

    print(Counter(txt_labels))

    classifier = "LogisticRegression"  # RidgeClassifier, SVC, LogisticRegression
    NaN_method = "zero"
    log_path = (
        "/home/ellie/Desktop/UAlberta/CMPUT605/Code/CMPUT605/logs/TwoClasses/MVOF_"
        + classifier
        + "_"
        + NaN_method
        + ".log"
    )

    logging.basicConfig(filename=log_path, level=logging.INFO)

    logging.info(f"Classifier: {classifier}\n NaN Method: {NaN_method}")

    all_subjects_confusion_matrices = []
    all_subjects_classification_reports = []

    for subj in subjects:
        logging.info(f"\n_______________________Subject: {subj}_______________________")

        # label_encoder = LabelEncoder()
        # labels = label_encoder.fit_transform(txt_labels)
        
        
        # labels = ilabels[indices]



        X_tmp, y_tmp = load_data(path=path, subject=subj, labels=mVof, method=NaN_method)

        
        ilabels  = ['F' if cls == 'LF' or cls == 'SM' else cls for cls in y_tmp]
        indices = [i for i, cls in enumerate(ilabels) if cls == 'F' or cls == 'M']
        
        y_c=[]
        for i in indices:
            y_c.append(ilabels[i] )

        X =   np.array([X_tmp[i] for i in indices if i < len(X_tmp)]  )
        y = np.array([0 if cls == 'F' else 1 for cls in y_c])
        print(Counter(y))
        return 0
        
        
        # print(Counter(y))
        
        

        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        desired_variance_ratio = 0.85
        pca = PCA(n_components=desired_variance_ratio)
        principal_components = pca.fit_transform(X_normalized)
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        num_components = (
            np.argmax(cumulative_variance_ratio >= desired_variance_ratio) + 1
        )
        selected_data = principal_components[:, :num_components]

        
        logging.info(
            f"\nNumber of components to explain {desired_variance_ratio * 100}% of variance: {num_components}"
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        subject_confusion_matrices = []
        subject_accuracies = []
        fold = 0

        for train_index, test_index in skf.split(selected_data, y):
            print(test_index)

            fold += 1
            logging.info(f"\n_______________________Fold {fold}_______________________")

            X_train, X_test = selected_data[train_index], selected_data[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print("\ny_test: "+ str(Counter(y_test)))
            print("y_train: "+str(Counter(y_train))+"\n")
            

            model = None
            if classifier == "LogisticRegression":
                model = LogisticRegression(penalty="l2", max_iter=1000)
                logging.info(
                    "\nLogisticRegression, l2 penalty for regularization, max iteration of 1000"
                )

            elif classifier == "RidgeClassifier":
                tmp_model = RidgeClassifier()
                param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
                grid_search = GridSearchCV(
                    estimator=tmp_model,
                    param_grid=param_grid,
                    scoring="neg_mean_squared_error",
                    cv=5,
                )
                grid_search.fit(X_train, y_train)
                best_alpha = grid_search.best_params_["alpha"]
                model = RidgeClassifier(alpha=best_alpha)
                logging.info(f"\nRidgeClassifier, best alpha {best_alpha}")

            elif classifier == "SVC":
                model = SVC(C=0.001, gamma=0.01, kernel="poly")
                param_grid = {
                    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    "kernel": ["poly"],  # , linear, "rbf", "poly", "sigmoid"],
                    "gamma": [0.01, 0.1, 1, 10.0, "scale", "auto"],
                }
                tmp_model = SVC()
                grid_search = GridSearchCV(
                    tmp_model, param_grid, cv=5, scoring="accuracy"
                )
                # logging.info(f"\nSVC, Best Params: {grid_search.best_params_} ")
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_

            model.fit(X_train, y_train)
            class_labels = model.classes_
            y_pred = model.predict(X_test)

            # conf_matrix = confusion_matrix(y_test, y_pred)
            conf_matrix_percent = confusion_matrix(y_test,y_pred,normalize='true')
            # conf_matrix_percent = (
            #     conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
            # ) * 100
            logging.info(f"\nConf Matrix:\n{conf_matrix_percent}")
            fold_accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"\nAccuracy: {fold_accuracy}")

            subject_accuracies.append(fold_accuracy)
            subject_confusion_matrices.append(conf_matrix_percent)

        # avg_conf_matrix = np.mean(subject_confusion_matrices, axis=0)
        d = np.zeros(shape=subject_confusion_matrices[0].shape)
        for i in subject_confusion_matrices:
            d = np.add(d,i)
        avg_conf_matrix = d / 5
        avg_classification_report = np.mean(subject_accuracies)
        logging.info("......................................................")
        logging.info(f"\n Subject Average Conf Matrix:\n{avg_conf_matrix}")
        logging.info(f"\n Subject Average Accuracy:\n{avg_classification_report}")

        all_subjects_confusion_matrices.append(avg_conf_matrix)
        all_subjects_classification_reports.append(avg_classification_report)

    # avg_conf_matrix_all_subjects = np.mean(all_subjects_confusion_matrices, axis=0)
    dd = np.zeros(shape=all_subjects_confusion_matrices[0].shape)
    for i in all_subjects_confusion_matrices:
        dd = np.add(dd,i)
    avg_conf_matrix_all_subjects = dd / len(all_subjects_confusion_matrices) 
    avg_classification_report_all_subjects = np.mean(
        all_subjects_classification_reports
    )
    logging.info("\n_______________________Final Report_______________________")

    # Print or use the results as needed
    logging.info(
        f"\nAverage Confusion Matrix Across All Subjects (Percentages): \n{avg_conf_matrix_all_subjects}"
    )
    logging.info(
        f"\nAverage Classification Report Across All Subjects: {avg_classification_report_all_subjects}"
    )

    # X_train, X_test, y_train, y_test = train_test_split(selected_data, y, test_size=0.2, random_state=42)

    sns.heatmap(
        avg_conf_matrix_all_subjects,
        annot=True,
        cmap="Blues",
        # xticklabels=label_encoder.classes_,
        # yticklabels=label_encoder.classes_,
        fmt='.2%'
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(
        "/home/ellie/Desktop/UAlberta/CMPUT605/Code/CMPUT605/logs/TwoClasses/"
        + "MVOF_avg_over_allsubjects_"
        + classifier
        + "_"
        + NaN_method
        + ".jpg"
    )
    plt.show()


if __name__ == "__main__":
    main()
