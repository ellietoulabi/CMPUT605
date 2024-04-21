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
from transformers import AutoTokenizer, AutoModel




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
            # print(arr_cleaned)
        elif method == "zero":
            flattened[np.isnan(flattened)] = 0
            arr_cleaned = flattened

        data.append(arr_cleaned)

    return np.array(data), np.array(labels)    


def sentence_embedding(sentences:list, method="bert"):
    """
    Args:
        sentences (list): list of sentence stimuli in the dataset
        method (str, optional): Method of snetence embedding. Values: 'bert', 'w2v'. Defaults to "bert".

    Returns:
        list: list of embeddings
    """    
    embeddings = []
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    for sentence in sentences:
        tokens = tokenizer(sentence, return_tensors="pt")
        outputs = model(**tokens)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        embeddings.append(embedding)
    
    return embeddings




from transformers import AutoTokenizer, AutoModel

def sentence_embedding2(sentences: list, method="bert", layer='middle'):
    """
    Generate embeddings for sentences using specified method and layer.
    
    Args:
        sentences (list): List of sentence stimuli in the dataset.
        method (str, optional): Method of sentence embedding. Supports 'bert', 'w2v'. Defaults to 'bert'.
        layer (str, optional): The layer from which to extract embeddings. Supports 'middle' for the middle layer,
            or an integer value specifying the exact layer. Defaults to 'middle'.
    
    Returns:
        list: List of embeddings.
    """
    embeddings = []
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    
    for sentence in sentences:
        tokens = tokenizer(sentence, return_tensors="pt")
        outputs = model(**tokens)
        
        # Decide which layer to use
        if layer == 'middle':
            layer_index = len(outputs.hidden_states) // 2  # Middle layer index
        elif isinstance(layer, int):
            layer_index = layer
        else:
            raise ValueError("Layer must be 'middle' or an integer.")
        
        # Extract embeddings from the specified layer
        embedding = outputs.hidden_states[layer_index].mean(dim=1).squeeze().detach().numpy()
        embeddings.append(embedding)
    
    return embeddings




def sentence_embeddingc(sentences:list, method="bert"):
    """
    Args:
        sentences (list): list of sentence stimuli in the dataset
        method (str, optional): Method of snetence embedding. Values: 'bert', 'w2v'. Defaults to "bert".

    Returns:
        list: list of embeddings
    """    
    embeddings = []
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    for sentence in sentences:
        tokens = tokenizer(sentence, return_tensors="pt")
        # embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        outputs = model(**tokens, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        cls = last_hidden_state[0,0,:].detach().numpy()
        embeddings.append(cls)
    
    return embeddings
