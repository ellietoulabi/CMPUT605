import numpy as np
from globals import txt_labels, sentence_stimuli, mVof
from functions import sentence_embedding, sentence_embedding2
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from collections import Counter


logname = "random_MVSM"
path = "/home/ellie/Desktop/UAlberta/CMPUT605/Code/CMPUT605/logs/Embed/"
log_path = path + logname + "_100.log"
logging.basicConfig(filename=log_path, level=logging.INFO)




# List of sentences and their labels
sentences = [
    "Some_desks_are_diamonds",
    "Some_desks_are_junkyards",
    "Some_hands_are_jungles",
    "Some_hands_are_magic",
    "Some_hearts_are_diseases",
    "Some_hearts_are_dwellings",
    "Some_hearts_are_ice",
    "Some_hearts_are_zoos",
    "Some_ideas_are_diamonds",
    "Some_ideas_are_gold",
    "Some_ideas_are_junkyards",
    "Some_ideas_are_snakes",
    "Some_jobs_are_jails",
    "Some_jobs_are_ribbons",
    "Some_lives_are_bulldozers",
    "Some_lives_are_tapestries",
    "Some_marriages_are_iceboxes",
    "Some_marriages_are_weapons",
    "Some_minds_are_closets",
    "Some_minds_are_daggers",
    "Some_rivers_are_jails",
    "Some_rivers_are_ribbons",
    "Some_roads_are_gold",
    "Some_roads_are_snakes",
    "Some_rumors_are_diseases",
    "Some_rumors_are_dwellings",
    "Some_salesmen_are_bulldozers",
    "Some_salesmen_are_tapestries",
    "Some_schools_are_ice",
    "Some_schools_are_zoos",
    "Some_smiles_are_butchers",
    "Some_smiles_are_razors",
    "Some_stores_are_jungles",
    "Some_stores_are_magic",
    "Some_surgeons_are_butchers",
    "Some_surgeons_are_razors",
    "Some_words_are_closets",
    "Some_words_are_daggers",
    "Some_words_are_iceboxes",
    "Some_words_are_weapons",
]

labels = [
    "SM",
    "M",
    "SM",
    "M",
    "SM",
    "M",
    "M",
    "SM",
    "M",
    "M",
    "SM",
    "SM",
    "M",
    "SM",
    "SM",
    "M",
    "M",
    "SM",
    "M",
    "SM",
    "SM",
    "M",
    "SM",
    "M",
    "M",
    "SM",
    "M",
    "SM",
    "SM",
    "M",
    "SM",
    "M",
    "M",
    "SM",
    "M",
    "SM",
    "SM",
    "M",
    "SM",
    "M",
]

similar = [[0, 1],
 [2, 3],
 [4, 5, 6, 7],
 [8, 9, 10, 11],
 [12, 13],
 [14, 15],
 [16, 17],
 [18, 19],
 [20, 21],
 [22, 23],
 [24, 25],
 [26, 27],
 [28, 29],
 [30, 31],
 [32, 33],
 [34, 35],
 [36, 37, 38, 39]]






import numpy as np
from sklearn.model_selection import KFold

# Function to generate 5-fold cross-validation sets with specific conditions
def generate_custom_cv_folds(sentences, similar_groups, n_splits=5):
    all_indexes = [index for group in similar_groups for index in group]
    
    np.random.shuffle(similar_groups)
    
    selection_conditions = [
        {'length_4': 2, 'length_2': 0},  # 2 arrays of length 4
        {'length_4': 0, 'length_2': 4},  # 4 arrays of length 2
        {'length_4': 1, 'length_2': 2}   # 1 array of length 4 and 2 arrays of length 2
    ]
    
    folds = []
    for _ in range(n_splits):
        test_indexes = []
        remaining_groups = similar_groups.copy()

        condition = np.random.choice(selection_conditions)
        
        for length, count in condition.items():
            length = int(length.split('_')[1])
            for _ in range(count):
                for group in remaining_groups:
                    if len(group) == length:
                        test_indexes.extend(group)
                        remaining_groups.remove(group)
                        break
        
        test_indexes = list(set(test_indexes))
        train_indexes = list(set(all_indexes) - set(test_indexes))
        folds.append((train_indexes, test_indexes))
    
    return folds

custom_cv_folds = generate_custom_cv_folds(sentences, similar)

# Print the first fold to check the structure
# print(custom_cv_folds)

embeddings = sentence_embedding2(sentences)
labels_t = np.array([0 if cls == 'SM' else 1 for cls in labels])
print("************************")
# print(labels_t)
import random

# random.shuffle(labels_t)
# random.shuffle(embeddings)
# print(labels_t)
print("************************")

hund_acc = []
hund_conf = []

# for i,j in zip(sentences,labels):
#     print(i,j)



for repeat in range(0,100):
    print(repeat)
    accuracies = []
    conf_matrices = []
    fold = 0
    for train_index, test_index in custom_cv_folds:
        print(train_index,test_index)
        fold += 1
        logging.info(f"\n_______________________Fold: {fold}_______________________")

        X_train, X_test = (
            np.array(embeddings)[train_index],
            np.array(embeddings)[test_index],
        )
        y_train, y_test = (
            np.array(labels_t)[train_index],
            np.array(labels_t)[test_index],
        )
        # random.shuffle(y_train)
        # random.shuffle(y_test)
        
        random.shuffle(X_train)
        random.shuffle(X_test)
        # Train a simple classifier, Logistic Regression in this case
        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        # clf = SVC.fit(X_train, y_train)

        # print(clf.classes_)
        # Predict and evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)




        pmax_stimuli_width = max(len(str(a)) for a in np.array(np.array(sentences))[test_index])
        pmax_labels_width = max(len(str(s)) for s in  np.array(np.array(labels_t))[test_index])
        
        # print(pmax_labels_width)
        # exit()

        # Print the table header
        print(f"{'Stimuli':<{pmax_stimuli_width}} | {'Actual':<{pmax_labels_width}} {'':<{pmax_labels_width}} | {'Predicted':<{pmax_labels_width}} ")
        # exit()
        # Print a separator
        # print(f"{'-' * pmax_stimuli_width}-+-{'-' * pmax_labels_width}-+-{'-' * pmax_labels_width}")
        print("===================================================")
        # exit()

        
        
        for a, s,t ,d in zip(np.array(np.array(sentences))[test_index], np.array(np.array(labels))[test_index], np.array(np.array(labels_t))[test_index], y_pred):
            print(f"{a:<{pmax_stimuli_width}} |  {s} ({t})  |  {d} ")
            
            
        print("_____________________________________________________________________________")
    
        # Confusion Matrix
        conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
        conf_matrices.append(conf_mat)

        logging.info(f"\nConf Matrix:\n{conf_mat}")
        logging.info(f"\nAccuracy: {acc}")

    # Calculate average accuracy
    average_accuracy = np.mean(accuracies)
    hund_acc.append(average_accuracy)

    # Calculate average confusion matrix
    # average_conf_matrix = np.mean(conf_matrices, axis=0)

    d = np.zeros(shape=conf_matrices[0].shape)
    for i in conf_matrices:
        d = np.add(d, i)
    avg_conf_matrix = d / 5
    
    hund_conf.append(avg_conf_matrix)


    print(f"Average Accuracy: {average_accuracy}")
    print("Average Confusion Matrix:")
    print(avg_conf_matrix)

    logging.info("\n_______________________Totlal_______________________")
    logging.info(f"\nConf Matrix:\n{average_accuracy}")
    logging.info(f"\nAccuracy: {avg_conf_matrix}")

final_accuracy = np.mean(hund_acc)

    # Calculate average confusion matrix
    # average_conf_matrix = np.mean(conf_matrices, axis=0)

fd = np.zeros(shape=hund_conf[0].shape)
for i in hund_conf:
    fd = np.add(fd, i)
final_conf_matrix = fd / 100
print("###############",final_accuracy)    
sns.heatmap(
    final_conf_matrix,
    annot=True,
    cmap="Blues",
    xticklabels=clf.classes_,
    yticklabels=clf.classes_,
    fmt=".2%",
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(path + logname + "_100.jpg")
plt.show()












