import torch
import infer
from utils import *
import argparse
from tqdm import tqdm
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy

def fine_tune(model, train_loader, train_loader_for_val, val_loader, optimal_centroids, epochs, lr):
    best_acc = infer.infer_with_KNN(model, train_loader_for_val, val_loader, 1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train(True)
        print_loss_for_epoch, print_loss_2, print_loss_3, print_batch_conv_div_loss, print_centroid_loss, print_loss_4, print_batch_conv_only_loss = (0, 0, 0, 0, 0, 0, 0) # for printing purpose only
        
        keep_batch_conv_div_loss = False
        keep_centroid_loss = True
        keep_centroid_loss_repulsion = True
        keep_batch_conv_only_loss = True


        loss_1 = 0
        loss_2 = 0
        loss_3 = 0
        loss_4 = 0

        for batch, (x, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            x, labels = x.to(device), labels.to(device)
            x_embeddings = model(x)
            #print(embeddings.shape)
            
            if(keep_batch_conv_div_loss):
                loss_1 = loss_batch_conv_div(x_embeddings, labels)


            if(keep_centroid_loss):
                loss_2 = centroid_loss_2(x_embeddings, labels, optimal_centroids)

            if(keep_centroid_loss_repulsion):
                loss_3 = centroid_loss_repulsion(x_embeddings, labels, optimal_centroids)

            if(keep_batch_conv_only_loss):
                loss_4 = loss_batch_conv_only(x_embeddings, labels)
            
            loss = loss_1 + 0.005*loss_2 + 0.005*loss_3 + loss_4

            loss.backward()
            optimizer.step()    
            
            # for printing purpose only
            print_loss_for_epoch += loss.item()
            print_loss_2 += loss_2.item()
            print_loss_3 += loss_3.item()
            print_loss_4 += loss_4.item()

        print(f'\nEpoch {epoch}') 
        print(f'total loss : {print_loss_for_epoch}')
        print(f'loss_2 (centroid convergence): {print_loss_2}')
        print(f'loss_3 (divergence from centroid): {print_loss_3}')
        print(f'loss_4 (batch_conv_only): {print_loss_4}')
        
        
        # Cross val
        cross_val_frequency = 1
        if(epoch % cross_val_frequency == 0):
            model.eval()

            curr_acc = infer.infer_with_KNN(model, train_loader_for_val, val_loader, 1)
            print('\nVal acc = ', curr_acc)
         
            if(curr_acc >= best_acc):
                save_path = f'saved_models/synthetic_loss_epoch_{epoch}.pth'
                torch.save(model.state_dict(), save_path)
                print(f'Model saved at {save_path}')
                best_acc = curr_acc
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 128)
    parser.add_argument('--epochs', type=int, default = 2)
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--val_gallery_folder', type=str)
    parser.add_argument('--val_data_folder', type=str)
    parser.add_argument('--lr', type=float, default=0.00001)
    
    
    args = parser.parse_args()
    
    # Synthetic data folders
    train_data_folder = args.train_data_folder
    
    # this is val_gallery
    for_val_train_data_folder = args.val_gallery_folder
    val_data_folder = args.val_data_folder
 
    train_loader, train_loader_for_val, val_loader  = utils_load_data(train_data_folder, for_val_train_data_folder, val_data_folder, args.batch_size)
    net = set_model()
        
    train_embeddings, train_targets = my_utils_get_embeddings(net, train_loader)
    initial_centroids = get_mean_embeddings(train_embeddings, train_targets)

    # Get optimal centroids
    copy_initial_centroids = copy.deepcopy(initial_centroids)
    optimal_centroids = get_optimal_virtual_centroids(copy_initial_centroids)

    fine_tune(net,train_loader, train_loader_for_val, val_loader, optimal_centroids, args.epochs, args.lr)
    
