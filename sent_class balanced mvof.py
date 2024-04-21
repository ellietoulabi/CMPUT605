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
from collections import Counter


logname = "MVOF_LR_balanced"
path = "/home/ellie/Desktop/UAlberta/CMPUT605/Code/CMPUT605/logs/Embed/"
log_path = path + logname + ".log"
logging.basicConfig(filename=log_path, level=logging.INFO)


embeddings_all = sentence_embedding(sentence_stimuli)


# encoder = OneHotEncoder(sparse=False)
# txt_labels_encoded = encoder.fit_transform(np.array(txt_labels).reshape(-1, 1))

labels = ["F" if cls == "LF" or cls == "SM"  else cls for cls in mVof]
indices = [i for i, cls in enumerate(labels) if cls == 'F' or cls == 'M']


labels_c = []
for i in indices:
    labels_c.append(labels[i] )
    
labels_o = []
for i in indices:
    labels_o.append(mVof[i] )
sentence_stimuli_t =   np.array([sentence_stimuli[i] for i in indices if i < len(sentence_stimuli)]  )

embeddings_t =   np.array([embeddings_all[i] for i in indices if i < len(embeddings_all)]  )
labels_t = np.array([0 if cls == 'F' else 1 for cls in labels_c])


# for i,j in zip(sentence_stimuli_t,labels_o):
#     print(i,j)
# exit()

indices_M = np.where(labels_t == 1)[0]

indices_O = np.where(labels_t == 0)[0]
random_indices_O = np.random.choice(indices_O, 20, replace=False)

selected_indices = np.concatenate((indices_M, random_indices_O))
    
print(selected_indices)

# indices_M = np.where(labels == 'M')[0]

# random_indices_O = np.random.choice(indices_O, 20, replace=False)
# print(np.array(random_indices_O))
# print(np.array(indices_M))



# selected_indices = np.concatenate((np.array(indices_M), np.array(random_indices_O)))
# print(np.array(embeddings_all).shape)

embeddings = np.array(embeddings_all)[selected_indices]


# txt_labels_encoded = np.array(
#     [0 if cls == "O" else 1 for cls in labels[selected_indices]]
# )


txt_labels_encoded=np.array(labels_t)[selected_indices]

# print(Counter(txt_labels_encoded))
# exit()

# Embed all sentences

# Prepare for Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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




    pmax_stimuli_width = max(len(str(a)) for a in np.array(np.array(sentence_stimuli_t)[selected_indices])[test_index])
    pmax_labels_width = max(len(str(s)) for s in  np.array(np.array(labels_o)[selected_indices])[test_index])
    
    # print(pmax_labels_width)
    # exit()

    # Print the table header
    print(f"{'Stimuli':<{pmax_stimuli_width}} | {'Actual':<{pmax_labels_width}} {'':<{pmax_labels_width}} | {'Predicted':<{pmax_labels_width}} ")
    # exit()
    # Print a separator
    # print(f"{'-' * pmax_stimuli_width}-+-{'-' * pmax_labels_width}-+-{'-' * pmax_labels_width}")
    print("===================================================")
    # exit()

    
    
    for a, s,t ,d in zip(np.array(np.array(sentence_stimuli_t)[selected_indices])[test_index], np.array(np.array(labels_o)[selected_indices])[test_index], np.array(np.array(labels_c)[selected_indices])[test_index], y_pred):
        print(f"{a:<{pmax_stimuli_width}} |  {s} ({t})  |  {d} ")
        
        
    print("_____________________________________________________________________________")
 
    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred, normalize="true")
    conf_matrices.append(conf_mat)

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
