import utils
from utils import *

import torch
import numpy as np
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_with_KNN(net, train_loader, test_loader, k):
    net.eval()
    # Infer train
    train_embeddings = np.zeros(shape=(0,128))
    train_targets = []

    for batch, (x,y) in enumerate(train_loader):
      x, y = x.to(device),  y.to(device)
      out = net(x)
      train_embeddings = np.concatenate([train_embeddings, out.detach().cpu().numpy()],axis=0)
      y = y.detach().cpu().tolist()
      train_targets.extend(y)

    print("Train embeddings shape : ",train_embeddings.shape)
    print("Targets : ", len(train_targets))

    # Infer test 
    test_embeddings = np.zeros(shape=(0,128))
    test_targets = []

    for batch, (x,y) in enumerate(test_loader):
      x, y = x.to(device),  y.to(device)

      out = net(x)
      test_embeddings = np.concatenate([test_embeddings, out.detach().cpu().numpy()],axis=0)
      y = y.detach().cpu().tolist()
      test_targets.extend(y)

    print("Test embeddings shape : ",test_embeddings.shape)
    print("Targets : ", len(test_targets))
    
	# Classify them
    for k_ in range(1, 2):
        accuracy = KNNAccuracy(train_embeddings, train_targets, test_embeddings, test_targets, k_)
        print('Test accuracy for k_ ', k_, accuracy)
    
    #accuracy = KNNAccuracy(train_embeddings, train_targets, test_embeddings, test_targets, k)
    return accuracy
    


# Use KNN to classify the embedding space
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=64)

    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels)

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)

    print(classification_report(test_labels, predictions))
    
    print('P, R, F for macro avg', precision_recall_fscore_support(test_labels, predictions, average='macro'))
    print('P, R, F for micro avg', precision_recall_fscore_support(test_labels, predictions, average='micro'))
    print('P, R, F for weighted avg', precision_recall_fscore_support(test_labels, predictions, average='weighted'))

    return (100 * accuracy_score(test_labels, predictions)) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", 
                        type=str)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument('--batch_size', default= 128, type=int) 
    parser.add_argument('--train_data_folder', type=str)  
    parser.add_argument('--test_data_folder', type=str)

    args = parser.parse_args()

    train_data_folder = args.train_data_folder
    test_data_folder = args.test_data_folder
    model_path = args.model_path

    train_loader, test_loader = utils_load_data_for_test(train_data_folder, test_data_folder, args.batch_size)

    # Trained model
    print("Trained model : ")
    net = set_model()

    net.load_state_dict(torch.load(args.model_path))
    
    net.eval()
    print('\nKNN accuracy')
    accuracy = infer_with_KNN(net, train_loader, test_loader, args.k)



