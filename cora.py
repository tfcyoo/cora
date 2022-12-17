#data manipulation
import numpy as np
import pandas as pd

#pytorch imports
import torch 
import torch.nn as nn
import torch.optim as optim

#random forest
from sklearn.ensemble import RandomForestClassifier

#Naives Bayes
from sklearn.naive_bayes import BernoulliNB

#dataviz
from matplotlib import pyplot as plt

#CV
from sklearn.model_selection import KFold

#manipulate files
import os



# dataviz: classes distribution in the dataset
def classes_distribution(cora):
    
    plt.hist(cora.iloc[:,cora.shape[1] - 1], density = True, bins = 7)
    plt.xticks(rotation = 90)
    plt.show()


def get_classes(cora):
    #last column index
    target_column = cora.shape[1] - 1

    #target classes 
    target_classes = cora.iloc[:, target_column].unique()
    print(target_classes)

    return target_classes


##### Encode target classes
def target_encode(cora):
    
    #target classes 
    target_classes = get_classes(cora)

    #create new target column at before last index 
    cora.insert(cora.shape[1] - 1, "target", 0)


    ### classes Encoding

    #Neural_Networks => 0
    #Rule_Learning => 1
    #Reinforcement_Learning => 2
    #Probabilistic_Methods => 3
    #Theory => 4
    #Genetic_Algorithms => 5
    #Case_Based => 6

    for i, target in enumerate(target_classes):
        cora.loc[cora.iloc[:, cora.shape[1]-1 ] == target, 'target'] = i

    return target_classes


####create MLP model with one hidden layer of size 64 and ReLU activation function
class PaperNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(PaperNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden = 128
        self.linear = nn.Linear(input_size, self.hidden)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.hidden, output_size)

    
    def forward(self, X):
        out = self.linear(X)
        out = self.relu(out)
        return self.linear_2(out)


# MLP training function
def train(paper_network, X_train, y, lr = 0.12, nber_iterations = 750):

    #cross entropy loss
    loss = nn.CrossEntropyLoss()

    #stochastic gradient descent optimizer
    sgd = optim.SGD(paper_network.parameters(), lr = lr)


    #training loop
    for i in range(nber_iterations):

        #forward pass
        y_pred = paper_network(X_train)

        #loss
        l = loss(y_pred, y)

        #backward
        l.backward()

        #update
        sgd.step()

        #reinitialize gradients
        sgd.zero_grad()
    

#model testing function
def test(paper_network, X_test):

     with torch.no_grad():

        y_pred_test = paper_network(X_test)

        _, y_pred_labels = torch.max(y_pred_test, dim = 1)

        return y_pred_labels



#Evaluation function
def accuracy(y_pred, y_test):

    n = len(y_test)
    exact_match_count = np.sum(y_test == y_pred)
    return exact_match_count/n


def write_result(papers_ids, predictions, target_classes, output):

    data_nn = {"paper_id": papers_ids, "prediction": target_classes[predictions]}
    data_nn = pd.DataFrame(data=data_nn)
    data_nn.to_csv(path_or_buf = output, sep= "\t",  mode='a', index = False, header = False)


def clean_result(accuracies):
    outputs = ["output_nn.csv", "output_rf.csv", "output_nb.csv"]
    max_index = accuracies.index(max(accuracies))

    for i in range(len(outputs)):

        if i != max_index:
            os.remove(outputs[i])

def training(X, y, rf, nb, target_classes):

    #initialising 10-fold cross-validation
    n_splits = 10
    kf = KFold(n_splits= n_splits, shuffle = True)


    #store models' accuracy 
    accuracy_nn = 0
    accuracy_rf = 0
    accuracy_nb = 0

    #clear output file if existing
    if os.path.exists("output_nn.csv"):
        os.remove("output_nn.csv")
    if os.path.exists("output_rf.csv"):
        os.remove("output_rf.csv")
    if os.path.exists("output_nb.csv"):
        os.remove("output_nb.csv")

    #train on gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #cross_val_score(paper_network, )
    nber_classes = len(target_classes)

    #cross-validation
    i = 1
    for train_idx, test_idx in kf.split(X):

        print(f'\n***Iteration {i} Test set Accuracy:')
        
        #make data usable by pytorch
        #train set
        X_train = torch.tensor(X.iloc[train_idx, 1: (X.shape[1]-1) ].values, dtype = torch.float32).to(device)
        y_train = torch.tensor(y.iloc[train_idx].values, dtype = torch.long).to(device)

        #test set
        X_test = torch.tensor(X.iloc[test_idx, 1: (X.shape[1]-1) ].values, dtype = torch.float32).to(device)
        y_test = torch.tensor(y.iloc[test_idx,].values, dtype = torch.long).to(device)
        papers_ids = X.iloc[test_idx, 0]

        
        #model training
        paper_network = PaperNetwork(X.shape[1]- 2, nber_classes).to(device)
        train(paper_network, X_train = X_train, y = y_train)  

        #model testing
        y_pred_labels = test(paper_network, X_test)  

        #model evaluation
        accuracy_nn += accuracy(y_pred_labels.cpu().numpy(), y_test.cpu().numpy())
        print(f'\t- NN: {accuracy(y_pred_labels.cpu().numpy(), y_test.cpu().numpy()):.3f}')

        #model training rf
        rf.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
        #model testing & evaluation rf
        rf_predictions = rf.predict(X_test.cpu().numpy())
        accuracy_rf += accuracy(rf_predictions, y_test.cpu().numpy())
        print(f'\t- RF: {accuracy(rf_predictions, y_test.cpu().numpy()):.3f}')


        #model training naive bayes
        nb.fit(X_train.cpu().numpy(), y_train.cpu().numpy())
        #model testing & evaluation rf
        nb_predictions = nb.predict(X_test.cpu().numpy())
        accuracy_nb += accuracy(nb_predictions, y_test.cpu().numpy())
        print(f'\t- NB: {accuracy(nb_predictions, y_test.cpu().numpy()):.3f}')
        
        write_result(papers_ids.values, y_pred_labels.cpu().numpy(), target_classes, "./output_nn.csv")
        write_result(papers_ids.values, nb_predictions, target_classes, "./output_rf.csv")
        write_result(papers_ids.values, y_pred_labels.cpu().numpy(), target_classes, "./output_nb.csv")
        
        i += 1

    
    print(f'\n\n**** MEAN ACCURACY')
    print(f'\t- NN: {accuracy_nn/n_splits:.3f}')
    print(f'\t- RF: {accuracy_rf/n_splits:.3f}')
    print(f'\t- NB: {accuracy_nb/n_splits:.3f}')
    clean_result([accuracy_nn, accuracy_rf, accuracy_nb])



def main():
    # read the data and print some of them

    #load data
    path_to_data = "./cora.content"
    cora = pd.read_csv(filepath_or_buffer= path_to_data, skiprows = 0, sep = "\t", header = None)

    #classes distribution
    #classes_distribution(cora)

    #encode target
    target_classes = target_encode(cora)
    
    #view data
    print(cora.head())

    #create random forest model
    rf = RandomForestClassifier(n_estimators= 1000, random_state=0, criterion= "gini", min_samples_split = 5)

    #Naive Bayes classifier
    nb = BernoulliNB(binarize = None)

    #target 
    y = cora["target"]

    #features
    X = cora.drop(cora.columns[cora.shape[1]-1], axis = 1)

    training(X, y, rf, nb, target_classes)


if __name__== "__main__":
  main()