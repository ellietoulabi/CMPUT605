import numpy as np
from globals import txt_labels,sentence_stimuli
from functions import sentence_embedding
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC








logname = "All_LR_unbalanced"
path =  "/home/ellie/Desktop/UAlberta/CMPUT605/Code/CMPUT605/logs/Embed/" 
log_path = (
    path
    + logname
    + ".log"
)
logging.basicConfig(filename=log_path, level=logging.INFO)


# encoder = OneHotEncoder(sparse=False)
# txt_labels_encoded = encoder.fit_transform(np.array(txt_labels).reshape(-1, 1))
txt_labels_encoded=txt_labels

# Embed all sentences
embeddings = sentence_embedding(sentence_stimuli)

# Prepare for Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
conf_matrices = []
fold = 0
for train_index, test_index in skf.split(embeddings, txt_labels_encoded):
    fold+=1
    logging.info(f"\n_______________________Fold: {fold}_______________________")

    
    X_train, X_test = np.array(embeddings)[train_index], np.array(embeddings)[test_index]
    y_train, y_test = np.array(txt_labels_encoded)[train_index], np.array(txt_labels_encoded)[test_index]
    
    # Train a simple classifier, Logistic Regression in this case
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    # clf = SVC.fit(X_train, y_train)

    # print(clf.classes_)
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred, normalize='true')
    conf_matrices.append(conf_mat)
    
    logging.info(f"\nConf Matrix:\n{conf_mat}")
    logging.info(f"\nAccuracy: {acc}")

# Calculate average accuracy
average_accuracy = np.mean(accuracies)

# Calculate average confusion matrix
# average_conf_matrix = np.mean(conf_matrices, axis=0)

d = np.zeros(shape=conf_matrices[0].shape)
for i in conf_matrices:
    d = np.add(d,i)
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
    fmt='.2%'
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(
    path
    + logname
    + ".jpg"
)
plt.show()

