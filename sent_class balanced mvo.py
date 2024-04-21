import numpy as np
from globals import txt_labels, sentence_stimuli, mVof
from functions import sentence_embedding
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


logname = "MVO_LR_balanced"
path = "/home/ellie/Desktop/UAlberta/CMPUT605/Code/CMPUT605/logs/Embed/"
log_path = path + logname + ".log"
logging.basicConfig(filename=log_path, level=logging.INFO)


embeddings_all = sentence_embedding(sentence_stimuli)


# encoder = OneHotEncoder(sparse=False)
# txt_labels_encoded = encoder.fit_transform(np.array(txt_labels).reshape(-1, 1))

labels = ["O" if cls == "LF" or cls == "SM" or cls == "LT" else "M" for cls in mVof]
# print(labels)

# indices_M = np.where(labels == 'M')[0]
indices_M = [
    25,
    58,
    60,
    61,
    66,
    67,
    79,
    94,
    98,
    104,
    110,
    112,
    113,
    115,
    118,
    122,
    126,
    128,
    155,
    159,
]

indices_O=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 62, 63, 64, 65, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 111, 114, 116, 117, 119, 120, 121, 123, 124, 125, 127, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158]
# indices_O = np.where(labels == "O")[0]
# print(indices_O)

random_indices_O = np.random.choice(indices_O, 20, replace=False)
# print(np.array(random_indices_O))
# print(np.array(indices_M))



selected_indices = np.concatenate((np.array(indices_M), np.array(random_indices_O)))
print(np.array(embeddings_all).shape)

embeddings = np.array(embeddings_all)[selected_indices]

# txt_labels_encoded = np.array(
#     [0 if cls == "O" else 1 for cls in labels[selected_indices]]
# )


txt_labels_encoded=np.array(labels)[selected_indices]
# from collections import Counter

# print(Counter(txt_labels_encoded))
# exit()
# Embed all sentences

# Prepare for Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=29)
accuracies = []
conf_matrices = []
fold = 0
for train_index, test_index in skf.split(embeddings, txt_labels_encoded):
    fold += 1
    logging.info(f"\n_______________________Fold: {fold}_______________________")

    X_train, X_test = (
        np.array(embeddings)[train_index],
        np.array(embeddings)[test_index],
    )
    y_train, y_test = (
        np.array(txt_labels_encoded)[train_index],
        np.array(txt_labels_encoded)[test_index],
    )

    # Train a simple classifier, Logistic Regression in this case
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    # clf = SVC.fit(X_train, y_train)

    # print(clf.classes_)
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
    conf_matrices.append(conf_mat)
    
    
    
    
    
    pmax_stimuli_width = max(len(str(a)) for a in np.array(np.array(sentence_stimuli)[selected_indices])[test_index])
    pmax_labels_width = max(len(str(s)) for s in  np.array(txt_labels_encoded)[test_index])
    
    # print(pmax_labels_width)
    # exit()

    # Print the table header
    print(f"{'Stimuli':<{pmax_stimuli_width}} | {'Actual':<{pmax_labels_width}} | {'Predicted':<{pmax_labels_width}}")
    # exit()
    # Print a separator
    # print(f"{'-' * pmax_stimuli_width}-+-{'-' * pmax_labels_width}-+-{'-' * pmax_labels_width}")
    print("===================================================")
    # exit()

    
    
    for a, s,t in zip(np.array(np.array(sentence_stimuli)[selected_indices])[test_index], np.array(txt_labels_encoded)[test_index], y_pred):
        print(f"{a:<{pmax_stimuli_width}} |  {s}  |  {t} ")
        
        
    print("_____________________________________________________________________________")
 
    
    
    

    logging.info(f"\nConf Matrix:\n{conf_mat}")
    logging.info(f"\nAccuracy: {acc}")

# Calculate average accuracy
average_accuracy = np.mean(accuracies)

# Calculate average confusion matrix
# average_conf_matrix = np.mean(conf_matrices, axis=0)

d = np.zeros(shape=conf_matrices[0].shape)
for i in conf_matrices:
    d = np.add(d, i)
avg_conf_matrix = d / 5


print(f"Average Accuracy: {average_accuracy}")
print("Average Confusion Matrix:")
print(avg_conf_matrix)

logging.info("\n_______________________Totlal_______________________")
logging.info(f"\nConf Matrix:\n{average_accuracy}")
logging.info(f"\nAccuracy: {avg_conf_matrix}")


sns.heatmap(
    avg_conf_matrix,
    annot=True,
    cmap="Blues",
    xticklabels=clf.classes_,
    yticklabels=clf.classes_,
    fmt=".2%",
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(path + logname + ".jpg")
plt.show()
