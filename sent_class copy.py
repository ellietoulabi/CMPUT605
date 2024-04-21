import numpy as np
from globals import txt_labels, sentence_stimuli
from functions import sentence_embedding
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import random
from collections import Counter


# Initialize logging
logname = "All_LR_balanced"
path = "/home/ellie/Desktop/UAlberta/CMPUT605/Code/CMPUT605/logs/Embed/"
log_path = path + logname + ".log"
logging.basicConfig(filename=log_path, level=logging.INFO)

# Balancing the dataset
unique_labels = set(txt_labels)
balanced_labels = []
balanced_stimuli = []

# Always include all 'M' and 'SM' labels
for label in ['M', 'SM']:
    for i, lbl in enumerate(txt_labels):
        if lbl == label:
            balanced_labels.append(lbl)
            balanced_stimuli.append(sentence_stimuli[i])

# Randomly pick 20 elements of 'LF' and 'LT'
for label in ['LF', 'LT']:
    indices = [i for i, lbl in enumerate(txt_labels) if lbl == label]
    selected_indices = random.sample(indices, 20)
    for i in selected_indices:
        balanced_labels.append(txt_labels[i])
        balanced_stimuli.append(sentence_stimuli[i])


# print(Counter(balanced_labels))
# for a,s in zip(balanced_stimuli,balanced_labels):
#     print(a,"   |   ",s)

max_stimuli_width = max(len(str(a)) for a in balanced_stimuli)
max_labels_width = max(len(str(s)) for s in balanced_labels)

# Print the table header
# print(f"{'Stimuli':<{max_stimuli_width}} | {'Labels':<{max_labels_width}}")

# Print a separator
# print(f"{'-' * max_stimuli_width}-+-{'-' * max_labels_width}")



# sorted_pairs = sorted(zip(balanced_stimuli, balanced_labels), key=lambda x: x[0])
# sorted_stimuli, sorted_labels = zip(*sorted_pairs)




# Print each row of the table
# for a, s in zip(sorted_stimuli, sorted_labels):
#     if(s in ["SM","M"]):
#         print(f"{a:<{max_stimuli_width}} | {s:<{max_labels_width}}")
#         # print(f"{a:<{max_stimuli_width}}")


# exit()
# Embed all sentences in the balanced dataset
embeddings = sentence_embedding(balanced_stimuli)

# Prepare for Stratified Cross Validation with the balanced dataset
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
conf_matrices = []
fold = 0

for train_index, test_index in skf.split(embeddings, balanced_labels):
    fold += 1
    logging.info(f"\n_______________________Fold: {fold}_______________________")
    
    X_train, X_test = np.array(embeddings)[train_index], np.array(embeddings)[test_index]
    y_train, y_test = np.array(balanced_labels)[train_index], np.array(balanced_labels)[test_index]
    
    # Train a simple classifier, Logistic Regression in this case
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    # print(X_test)
    # exit()
    
    
    
    pmax_stimuli_width = max(len(str(a)) for a in np.array(balanced_stimuli)[test_index])
    pmax_labels_width = max(len(str(s)) for s in  np.array(balanced_labels)[test_index])
    
    # print(pmax_labels_width)
    # exit()

    # Print the table header
    print(f"{'Stimuli':<{pmax_stimuli_width}} | {'Actual':<{pmax_labels_width}} | {'Predicted':<{pmax_labels_width}}")
    # exit()
    # Print a separator
    # print(f"{'-' * pmax_stimuli_width}-+-{'-' * pmax_labels_width}-+-{'-' * pmax_labels_width}")
    print("===================================================")
    # exit()

    
    
    for a, s,t in zip(np.array(balanced_stimuli)[test_index], np.array(balanced_labels)[test_index], y_pred):
        print(f"{a:<{pmax_stimuli_width}} |  {s}  |  {t} ")
        
        
    print("_____________________________________________________________________________")
 
    
    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred, normalize='true')
    conf_matrices.append(conf_mat)
    
    logging.info(f"\nConf Matrix:\n{conf_mat}")
    logging.info(f"\nAccuracy: {acc}")

# Calculate average accuracy and average confusion matrix
average_accuracy = np.mean(accuracies)
avg_conf_matrix = np.sum(conf_matrices, axis=0) / len(conf_matrices)

print(f"Average Accuracy: {average_accuracy}")
print("Average Confusion Matrix:")
print(avg_conf_matrix)

logging.info("\n_______________________Total_______________________")
logging.info(f"\nConf Matrix:\n{avg_conf_matrix}")
logging.info(f"\nAccuracy: {average_accuracy}")

# Plotting
sns.heatmap(
    avg_conf_matrix,
    annot=True,
    cmap="Blues",
    xticklabels=clf.classes_,
    yticklabels=clf.classes_,
    fmt=".2%"
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(path + logname + ".jpg")
plt.show()
