from torchvision import datasets, transforms
import torch
import numpy as np
import os
from PIL import Image
#from networks.resnet_big import *
# from networks.my_resnet_pretrained import *
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import accuracy_score
import infer
from statistics import mean, median
from models.resnet import Orig_resnet_with_projection


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ClassSpecificImageFolder(datasets.DatasetFolder):
    # Source: https://discuss.pytorch.org/t/using-only-some-of-subfolders-in-dataset/72323/11#:~:text=I%20had%20a,standard%20ImageFolder%20class%3A
    def __init__(
            self,
            root,
            dropped_classes=[],
            transform = None,
            target_transform = None,
            loader = datasets.folder.default_loader,
            is_valid_file = None,
    ):
        self.dropped_classes = dropped_classes
        super(ClassSpecificImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                                       transform=transform,
                                                       target_transform=target_transform,
                                                       is_valid_file=is_valid_file)
        self.imgs = self.samples

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [c for c in classes if c not in self.dropped_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

class Custom_resize_transform(object):
    def __init__(self, output_size = (224, 224), mean = (0.4404, 0.4681, 0.4242), std = (0.3130, 0.3067, 0.3141)):
        #assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.mean = mean
        self.std = std

    def __call__(self, img):

        old_size = img.size # width, height
        ratio = float(self.output_size[0])/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)
        # Paste into centre of black padded image
        new_img = Image.new("RGB", (self.output_size[0],self.output_size[1]))
        new_img.paste(img, ((self.output_size[0]-new_size[0])//2, (self.output_size[1]-new_size[1])//2))
        
        return new_img


def my_utils_load_data(train_data_folder, test_data_folder, dropped_classes, batch_size):
    print('\nLoading data (my_utils_load_data)')
    mean = (0.4404, 0.4681, 0.4242)
    std = (0.3130, 0.3067, 0.3141)

    normalize = transforms.Normalize(mean=mean, std=std)

    transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            Custom_resize_transform(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = ClassSpecificImageFolder(root=train_data_folder, dropped_classes = dropped_classes, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

    train_dataset_for_test = ClassSpecificImageFolder(root=train_data_folder, dropped_classes = [], transform=transform)
    test_dataset = ClassSpecificImageFolder(root=test_data_folder, dropped_classes = [], transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True)

    train_loader_for_test = torch.utils.data.DataLoader(train_dataset_for_test, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader,val_loader,train_loader_for_test, test_loader


def utils_load_data(train_data_folder, for_val_train_data_folder, val_data_folder, batch_size):
    mean=(0.2032, 0.1978, 0.2084) # for synthetic data
    std=(0.3502, 0.3455, 0.3491) # for synthetic data
    dropped_classes = []
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_train = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.6, saturation=0.8, hue=0.1)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], p=0.7),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.6),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.01, 2))], p=0.7),
            Custom_resize_transform(),
            transforms.ToTensor(),
            normalize,
        ])
    
    transfor_test = transforms.Compose([
            Custom_resize_transform(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = ClassSpecificImageFolder(root=train_data_folder, dropped_classes = dropped_classes, transform=transform_train)
    train_dataset_for_val = ClassSpecificImageFolder(root=for_val_train_data_folder, dropped_classes = dropped_classes, transform=transfor_test)
    val_dataset = ClassSpecificImageFolder(root=val_data_folder, dropped_classes = dropped_classes, transform=transfor_test)
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset_for_val, batch_size = batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True)
    
    return train_loader, train_loader_for_val, val_loader

def utils_load_data_for_test(train_data_folder, test_data_folder, batch_size):
    mean=(0.2032, 0.1978, 0.2084) # for synthetic data
    std=(0.3502, 0.3455, 0.3491) # for synthetic data
    dropped_classes = []
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_test = transforms.Compose([
            Custom_resize_transform(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = ClassSpecificImageFolder(root=train_data_folder, dropped_classes = dropped_classes, transform=transform_test)
    test_dataset = ClassSpecificImageFolder(root=test_data_folder, dropped_classes = dropped_classes, transform=transform_test)
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=True)
    
    return train_loader, test_loader



def my_utils_get_embeddings(net, data_loader):

    embeddings = np.zeros(shape=(0,128))
    targets = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch, (x,y) in enumerate(data_loader):
      x, y = x.to(device),  y.to(device)
      out = net(x)
      embeddings = np.concatenate([embeddings, out.detach().cpu().numpy()],axis=0)
      y = y.detach().cpu().tolist()
      targets.extend(y)

    print("Data embeddings shape : ", embeddings.shape)
    print("Targets : ", len(targets))
    return torch.tensor(embeddings), torch.tensor(targets)

def get_mean_embeddings(embeddings, targets):
    N, d = embeddings.size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_classes = list(set(targets.tolist()))
    print('\nClasses in this dataset: ', all_classes)
    
    class_mean = torch.zeros(max(all_classes) + 1, d).to(device)
    for c in all_classes:
        class_mean[c] = (embeddings[targets == c].mean(0))

    class_mean = list(class_mean)
    for c in class_mean:
        c.requires_grad_()

    print('\nInitial distances : \n')
    for c in class_mean:
        print('\n')
        for c_ in class_mean:
            print(torch.nn.PairwiseDistance(p=2)(c, c_))

    print('\nInitial dot products : \n')
    for c in class_mean:
        print('\n')
        for c_ in class_mean:
            print(torch.dot(c, c_))

    return class_mean

def analyze(class_mean):
    dot_products = {}
    num_classes = len(class_mean)
    for c in range(num_classes):
        dot_products[c] = []
        for c_ in range(num_classes):
            if(c != c_):
                dot_products[c].append(torch.dot(class_mean[c], class_mean[c_]).item())
    #Analysis
    print("=========================")
    print("Centroids Analysis....\n")
    print("=========================")
    for c in range(num_classes):
        print(f"Class #{c} : ")
        print(f"\tMean \t:: {mean(dot_products[c])}")
        print(f"\tMedian \t:: {median(dot_products[c])}")
        print(f"\tMax \t:: {max(dot_products[c])}")
        print(f"\tMin \t:: {min(dot_products[c])}")
        '''
        print(f"\tMax \t:: {max(dot_products[c])}\t dot_products_index \t:: {dot_products[c].index(max(dot_products[c]))}")
        print(f"\tMin \t:: {min(dot_products[c])}\t dot_products_index \t:: {dot_products[c].index(min(dot_products[c]))}")
        '''

def get_optimal_virtual_centroids(class_mean):
    print('Finding optimal virtual centroids...')
    pdist = torch.nn.PairwiseDistance(p=2)
    epochs = 200
    lr = 0.1
    #print('Loss : ')
    for epoch in range(epochs):
        loss = 0
        for c in class_mean:
            for c_ in class_mean:
                if(not torch.equal(c, c_)):
                    loss -= pdist(c, c_)
        
        loss.backward()
        with torch.no_grad():
             for c in class_mean:
                c -= lr * c.grad
                c /= torch.norm(c)
                c.grad = None
        if(epoch % 20 ==0):
            print(loss)
    return class_mean

def loss_batch_conv_div(x_embeddings, labels):
    dist = ((x_embeddings.unsqueeze(1) - x_embeddings.unsqueeze(0)) ** 2).mean(-1)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    mask = mask - torch.diag(torch.diag(mask))
    neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    loss_1 = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) - (dist  * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
    loss_1 = loss_1.mean()
    return loss_1

def loss_batch_conv_only(x_embeddings, labels):
    dist = ((x_embeddings.unsqueeze(1) - x_embeddings.unsqueeze(0)) ** 2).mean(-1)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    mask = mask - torch.diag(torch.diag(mask))
    loss_4 = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3)
    loss_4 = loss_4.mean()
    return loss_4

def centroid_loss(x_embeddings, labels, optimal_centroids):    
    # loss 2
    pdist = torch.nn.PairwiseDistance(p=2)
    num_labels = len(set(labels.tolist()))
    #print(num_labels, labels)
    N, d = x_embeddings.size()
    class_mean = torch.zeros(num_labels, d).cuda()

    for c in range(num_labels):
        this_class_elements =  x_embeddings[labels.clone().detach() == c]
        if(this_class_elements.numel()):
            class_mean[c] = this_class_elements.mean(0)

    loss_2 = 0
    for n in range(num_labels):
        if(not torch.equal(class_mean[n], torch.zeros(d).to(device))): #to ensure that we have at least one data point for a label
            #print(n)
            distance_ = pdist(class_mean[n], optimal_centroids[n])
            #print(distance_)
            loss_2 += distance_
    loss_2 /= num_labels
    return loss_2

def centroid_loss_2(x_embeddings, labels, optimal_centroids):
    # loss 2
    pdist = torch.nn.PairwiseDistance(p=2)
    loss = 0
    for i in range(len(x_embeddings)):
        distance_ = pdist(x_embeddings[i], optimal_centroids[labels[i].clone().detach()])
        loss += distance_

    loss /= len(x_embeddings)
    return loss

def centroid_loss_repulsion(x_embeddings, labels, optimal_centroids):
    # loss 2
    pdist = torch.nn.PairwiseDistance(p=2)
    loss = 0
    for i in range(len(x_embeddings)):
        for j in range(len(optimal_centroids)):
            if j != labels[i].clone().detach():
                distance_ = pdist(x_embeddings[i], optimal_centroids[j])
                loss += 1 / distance_

    loss /= (len(x_embeddings) - 1)
    loss /= len(x_embeddings)
    return loss

def prepare_ood(model, dataloader):
        bank = None
        label_bank = None
        for x, labels in dataloader:
            x, labels = x.to(device), labels.to(device)
            model.eval()
            pooled = model(x)

            if bank is None:
                bank = pooled.clone().detach()
                label_bank = labels.clone().detach()
            else:
                bank_local = pooled.clone().detach()
                label_bank_local = labels.clone().detach()
                bank = torch.cat([bank_local, bank], dim=0)
                label_bank = torch.cat([label_bank_local, label_bank], dim=0)

        #self.norm_bank = F.normalize(self.bank, dim=-1)
        N, d = bank.size()
        all_classes = list(set(label_bank.tolist()))
        class_mean = torch.zeros(max(all_classes) + 1, d).to(device)
        for c in all_classes:
            class_mean[c] = (bank[label_bank == c].mean(0))
        centered_bank = (bank - class_mean[label_bank]).detach().cpu().numpy()
        precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(np.float32)
        class_var = torch.from_numpy(precision).float().to(device)
        return class_mean, class_var

def get_val_acc_and_loss(model, train_loader, test_loader, optimal_centroids):
    acc = infer_with_maha.infer_with_maha(model, train_loader, test_loader)
    
    conv_div_loss, controid_loss = (0, 0)
    for batch, (x,labels) in enumerate(test_loader):
        x, labels = x.to(device),  labels.to(device)
        out = model(x)
        # Finding loss
        conv_div_loss += loss_batch_conv_div(out, labels).item()
        controid_loss += centroid_loss_2(out, labels, optimal_centroids).item()
    

    #conv_div_loss /= batch
    #controid_loss /= batch
    overall_loss = conv_div_loss + controid_loss
    return acc, overall_loss, conv_div_loss, controid_loss

def get_val_loss(model, test_loader, optimal_centroids):
    conv_div_loss, batch_conv_loss, centroid_conv_loss, centroids_div_loss = (0, 0, 0, 0)
    for batch, (x,labels) in enumerate(test_loader):
        x, labels = x.to(device),  labels.to(device)
        out = model(x)
        # Finding loss
        conv_div_loss += loss_batch_conv_div(out, labels).item()
        batch_conv_loss += loss_batch_conv_only(out, labels).item()
        centroid_conv_loss += centroid_loss_2(out, labels, optimal_centroids).item()
        centroids_div_loss += centroid_loss_repulsion(out, labels, optimal_centroids).item()
    

    #conv_div_loss /= batch
    #controid_loss /= batch
    weighted_overall_loss = batch_conv_loss + 0.005*centroid_conv_loss + 0.005*centroids_div_loss
    return conv_div_loss, batch_conv_loss, centroid_conv_loss, centroids_div_loss, weighted_overall_loss


def get_data_size(dataloader):
    total_images = 0
    total_labels = 0

    for batch in dataloader:
        images, labels = batch
        print(type(images), type(labels))
        # total_images += images.size(0)  # Add the number of images in the batch
        # total_labels += labels.size(0)  # Add the number of labels in the batch
        total_images += len(images)  # Add the number of images in the batch
        total_labels += len(labels)  # Add the number of labels in the batch

    print(f"Total Images: {total_images}")
    print(f"Total Labels: {total_labels}")


def save_model(model, optimizer, opt, epoch, save_file):
    print('\n==> Saving at ', save_file)
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_model():
    model = Orig_resnet_with_projection()

    if torch.cuda.is_available():
        """
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        """
        model = model.cuda()

    
    return model