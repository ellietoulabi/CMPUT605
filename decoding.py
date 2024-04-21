import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# from transformers import AutoTokenizer, AutoModel
from globals import txt_labels,sentence_stimuli
from functions import load_data,sentence_embedding
from scipy.stats import pearsonr





def main():

    path = "/home/ellie/Desktop/UAlberta/CMPUT605/Code/Gonzo/"


    NaN_method = "drop"

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(txt_labels)

    X, y = load_data(path=path, subject="P055", labels=labels, method=NaN_method)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    desired_variance_ratio = 0.85
    pca = PCA(n_components=desired_variance_ratio)
    principal_components = pca.fit_transform(X_normalized)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance_ratio >= desired_variance_ratio) + 1
    selected_data = principal_components[:, :num_components]
    
    
    # Sentence Embedding
    embeddings = np.array(sentence_embedding(sentence_stimuli, method='bert'))
    
    
    
    
    kf = KFold(n_splits=embeddings.shape[0], shuffle=True, random_state=42)
    subject_mses = []
    subject_avg_correlations = []
    fold = 0

    for train_index, test_index in kf.split(selected_data, embeddings):

        fold += 1

        X_train, X_test = selected_data[train_index], selected_data[test_index]
        y_train, y_test = y[train_index], y[test_index]
     
        
        ridge_model = Ridge(alpha=0.1)
        ridge_model.fit(X_train, y_train)

        y_pred = ridge_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        correlations = []
        for true_row, pred_row in zip(y_test, y_pred):
            corr, _ = pearsonr(true_row, pred_row)
            correlations.append(corr)
        print(f"____________________________________ Fold {fold} ____________________________________")
        print("Mean Squared Error:", mse)
        print("Average Correlation:" , np.mean(correlations))
        subject_avg_correlations.append(np.mean(correlations))
        subject_mses.append(mse)
        
    
   
    
    
    
    
    
    
    # X_train, X_test, y_train, y_test = train_test_split(selected_data, embeddings, test_size=0.2, random_state=42)

    # Create and train the Ridge Regression model
    # ridge_model = Ridge(alpha=1.0)  # You can adjust the regularization strength (alpha) as needed
    # ridge_model.fit(X_train, y_train)

    # # Predict on the test set
    # y_pred = ridge_model.predict(X_test)

    # # Evaluate the model
    # mse = mean_squared_error(y_test, y_pred)
    # print("Mean Squared Error:", mse)
    # print(y_pred.shape)
    
    
    

    # correlations = []

    # # Iterate over each row in y_test and y_pred
    # for true_row, pred_row in zip(y_test, y_pred):
    #     corr, _ = pearsonr(true_row, pred_row)
    #     correlations.append(corr)

    # print("Correlation of each row in test with pred:")
    # for i, corr in enumerate(correlations, 1):
    #     print(f"Row {i}: {corr}")
        
    # print("Average:" , np.mean(correlations))
        
    
   
    
    
    
    


if __name__ == "__main__":
    main()



# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")

# s = "Some birds are eagles"
# tokens = tokenizer(s, return_tensors="pt")
# outputs = model(**tokens)

# em = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# print(em.shape)
