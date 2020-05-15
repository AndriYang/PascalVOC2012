import pandas as pd
import os
from bs4 import BeautifulSoup
#from more_itertools import unique_everseen
import numpy as np
#import matplotlib.pyplot as plt
#import skimage
#from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image

import matplotlib.pyplot as plt 

import sklearn
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import shutil

import random

class PascalVOC:
    """
    Handle Pascal VOC dataset
    """
    def __init__(self, root_dir, dataset, transform=None):
        """
        Summary: 
            Init the class with root dir
        Args:
            root_dir (string): path to your voc dataset
        """
        self.root_dir = root_dir
        self.img_dir =  os.path.join(root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root_dir, 'csvs')
        self.transform = transform
        self.image_paths, self.labels = self.get_data_multilabel(dataset)
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def list_image_sets(self):
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def _imgs_from_category(self, cat_name, dataset):
        """
        Summary: 
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary: 
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        df = df[df['true'] == 1]
        return df['filename'].values
    
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # apply transforms 
        image_path = self.image_paths[idx] 
        image = self.get_image(image_path)
        if self.transform is not None:
            image = self.transform(image)
        label = (self.labels[idx])
        label =torch.from_numpy(label)
        return [image, label, image_path]
#    
    def get_image(self, image_path):
        img = Image.open(image_path)
        img.load()
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, 2)
        return Image.fromarray(img)
#    
    
    def _annotation_file_from_img(self, img_name):
        
        path = os.path.join(self.ann_dir, img_name) + '.xml'
        
        return path
    
    def load_annotation(self, img_filename):
        
        xml = ""
        with open(self._annotation_file_from_img(img_filename)) as file:
            xml = file.readlines()
        xml = ''.join([line.strip("\t") for line in xml])
        return BeautifulSoup(xml, "xml")

    def get_data_multilabel(self, type_of_data):
        
        filename = os.path.join(self.set_dir, type_of_data + ".txt")
        category_list = self.list_image_sets()
        df = pd.read_csv(filename, sep=' ', header=None, names=["filename"])
        
        for category_name in category_list:
            df[category_name] = 0
        for data in df.itertuples():
            ind, fname = data[0], data[1]
            annotation = self.load_annotation(fname)
            objects = annotation.findAll("object")
            for object in objects:
                object_names = object.findChildren("name")
                for name_tag in object_names:
                    tag_name = name_tag.contents[0]
                    tag_name = str(tag_name)
                    if tag_name in category_list:
                        df.at[ind, tag_name] = 1
        df["filename"] = self.img_dir + df['filename'] + ".jpg"
        df_filename = df["filename"]
        df_label = df.drop(['filename'], axis=1)
        return df_filename, df_label.to_numpy()
    
    
def plot_loss_graph(epoch, training_losses, validation_losses, init_folder):
    epoch_list = np.arange(1, epoch + 1)
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_list, training_losses, label = "Training loss")
    plt.plot(epoch_list, validation_losses, label = "Validation loss")
    plt.legend(loc = "upper right")
    path = str(init_folder) + "/loss.png"
    plt.savefig(path)
    plt.show()

def plot_accuracy_graph(epoch, training_accuracy, validation_accuracy, init_folder):
    epoch_list = np.arange(1, epoch + 1)
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Average Precision")
    plt.plot(epoch_list, training_accuracy, label = "Training MAP")
    plt.plot(epoch_list, validation_accuracy, label = "Validation MAP")
    plt.legend(loc = "lower right")
    path = str(init_folder) + "/MAP.png"
    plt.savefig(path)
    plt.show()

def plot_tailacc_graph(epoch, tail_accuracy, init_folder):
    tail_accuracy = np.array(tail_accuracy)
    ave_tail_accs = np.mean(tail_accuracy, axis=0)
    plt.xticks(epoch)
    plt.xlabel("t")
    plt.ylabel("Tail Accuracy")
    plt.plot(epoch, ave_tail_accs, label = "Tail Accuracy")
    plt.legend(loc = "lower right")
    path = str(init_folder) + "/tailacc.png"
    plt.savefig(path)
    plt.show()
    
def AP(y_true, y_pred, n_classes=20):
    average_precision = dict()
    for i in range(n_classes):
        average_precision[i] = average_precision_score( np.array(torch.flatten(y_true[:, i].cpu())),  np.array(torch.flatten(y_pred[:, i].cpu())))
    return average_precision

def get_x_catagories_index(num_categories):
    cat_indx_list = []
    while len(cat_indx_list) < num_categories:
        x= random.randrange(0,len(dataset_test.list_image_sets()))
        if x not in cat_indx_list:
            cat_indx_list.append(x)
        else:
            continue
    return cat_indx_list

def get_top_x_imgs_categories(cat_indx_list, num_imgs, init_folder):
    for i in cat_indx_list:
        group = images_path_sorted[i]
        category = dataset_test.list_image_sets()[i]
        path = str(init_folder) + "/Top " + str(num_imgs) +  "_" + str(category)
        if not os.path.exists(path):
            os.makedirs(path)
        print('Top 5 for Category ', category)
        for s in range(num_imgs):
            img_path = path +"/image_{}.png".format(s)
            img = mpimg.imread(group[s],0)
            plt.imshow(img)
            print ('saving image: ', img_path)
            plt.savefig(img_path)
            plt.show()
            
def get_last_x_imgs_categories(cat_indx_list, num_imgs, init_folder):
    for i in cat_indx_list:
        group = images_path_sorted[i]
        category = dataset_test.list_image_sets()[i]
        path = str(init_folder) + "/Last_" + str(num_imgs) +  "_" + str(category)
        if not os.path.exists(path):
            os.makedirs(path)
        print('Last 5 for Category ', category)
        for s in range(len(group)-num_imgs,len(group)):
            img_path = path +"/image_{}.png".format(s)
            img = mpimg.imread(group[s],0)
            plt.imshow(img)
            print ('saving image: ', img_path)
            plt.savefig(img_path)
            plt.show()
            
def get_rank_by_categories(pred_scores, imgs_path, col=0):
    score = np.copy(pred_scores)
    img_dir = imgs_path.copy()
    order = score[:,col]
    order = order.argsort()[::-1]
    img_dir=np.array(img_dir)
    return score[order], order, img_dir[order]

def get_individual_tail_acc(sorted_pred_scores, sorted_label, t_min, t_max, t_num):
    t_vals = np.linspace(t_min, t_max, t_num, endpoint=False)
    tail_accs = []
    for t_val in t_vals:
        tp = 0
        fp = 0
        for (score, label) in zip(sorted_pred_scores, sorted_label):
            pred = 0
            if t_val < score:
                pred = 1
                if pred != label:
                    fp += 1
                else:
                    tp += 1
                    
        tail_acc = 0
        if fp + tp > 0:
            tail_acc = tp / (fp + tp)
        tail_accs.append(tail_acc)
    return tail_accs

def get_tail_acc(pred_scores, labels, img_paths, t_min=0.5, t_num=10):
    t_max = np.max(pred_scores)
    tail_accs = []
    t_vals = np.linspace(t_min, t_max, t_num, endpoint=False)
    for i in range(20):
        sorted_pred_scores, idx_order, sorted_imgs_path = get_rank_by_categories(pred_scores=pred_scores, imgs_path=img_paths, col=i)
        first_50_pred_scores = sorted_pred_scores[:50,i]
        first_50_labels = labels[idx_order][:50,i]
        tail_accs_i = get_individual_tail_acc(first_50_pred_scores, first_50_labels, t_min, t_max, t_num)
        tail_accs.append(tail_accs_i)
    return tail_accs, t_vals
    
def train_epoch(model, trainloader, criterion, device, optimizer, epoch, batch_size):
  
    model.train()
    losses = list()
    for batch_idx, data in enumerate(trainloader):
        inputs=data[0].to(device)
        labels=data[1].long().to(device)
        img_paths = data[2]
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.type_as(outputs)
        preds= torch.sigmoid(outputs)
        assert len(data[1]) == len(preds), 'different length'
        if batch_idx == 0:
            pred_scores = preds
            targets = labels
            imgs_path = list(img_paths)
        else:
            pred_scores = torch.cat((pred_scores, preds))
            targets = torch.cat((targets, labels))
            imgs_path = imgs_path +  list(img_paths)
            
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    loss = np.mean(losses)
    avg_precision = AP(targets, pred_scores.detach())
    avg_precision = [ 0 if np.isnan(i) else i for i in list(avg_precision.values())]
    mean_avg_precision = np.mean(avg_precision)
    return loss, avg_precision, mean_avg_precision

def evaluate_val(model,val_loader, criterion, device, init_folder):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for ctr, data in enumerate(val_loader):
            inputs = data[0].to(device)
            labels = data[1].long().to(device)
            img_paths = data[2]
            outputs = model(inputs)
            labels = labels.type_as(outputs)
            val_loss += criterion(outputs, labels).item()
            preds= torch.sigmoid(outputs)
            assert len(data[1]) == len(preds), 'different length'
            if ctr == 0:
                pred_scores = preds
                targets = labels
                imgs_path = list(img_paths)
            else:
                pred_scores = torch.cat((pred_scores, preds))
                targets = torch.cat((targets, labels))
                imgs_path = imgs_path +  list(img_paths)
    avg_precision = AP(targets, pred_scores)
    print('avg_precision: ', avg_precision)
    with open(init_folder + '/val_avg_precisions.txt', 'w') as the_file:
        the_file.write(str(avg_precision))
    val_loss /= len(val_loader)
    avg_precision = [ 0 if np.isnan(i) else i for i in list(avg_precision.values())]
    mean_avg_precision = np.mean(avg_precision)
    print('mean_avg_precision',mean_avg_precision)
    
    return val_loss, avg_precision, mean_avg_precision

def evaluate_test(model,test_loader, criterion, device):

    model.eval()
    
    with torch.no_grad():
        for ctr, data in enumerate(test_loader):
            inputs = data[0].to(device)
            labels = data[1].long().to(device)
            img_paths = data[2]
            outputs = model(inputs)
            preds= torch.sigmoid(outputs)
            assert len(data[1]) == len(preds), 'different length'
            if ctr == 0:
                pred_scores = preds
                targets = labels
                imgs_path = list(img_paths)
            else:
                pred_scores = torch.cat((pred_scores, preds))
                targets = torch.cat((targets, labels))
                imgs_path = imgs_path +  list(img_paths)
    avg_precision = AP(targets, pred_scores)
    avg_precision = [ 0 if np.isnan(i) else i for i in list(avg_precision.values())]
    mean_avg_precision = np.mean(avg_precision)
    
    return  avg_precision, mean_avg_precision, pred_scores, targets, imgs_path

def train_modelcv(dataloader_cvtrain, dataloader_cvVal ,  model, criterion, optimizer, scheduler, num_epochs, batch_size, device, init_folder):

    v_losses=[]
    v_avg_precisions=[]
    v_mean_avg_precisions=[]
    train_losses=[]
    train_avg_precisions=[]
    train_mean_avg_precisions=[]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
            
        model.train(True)
        train_loss,train_avg_precision, train_mean_avg_precision=train_epoch(model,  dataloader_cvtrain, criterion,  device , optimizer, epoch, batch_size)
        train_losses.append(train_loss)
        train_avg_precisions.append(train_avg_precision)
        train_mean_avg_precisions.append(train_mean_avg_precision)
        
        model.train(False)
        v_loss, v_avg_precision, v_mean_avg_precision= evaluate_val(model, dataloader_cvVal, criterion, device, init_folder)
        
        if (len(v_losses) > 0) and (v_loss < min(v_losses)):
            torch.save(model.state_dict(), "model_parameter.pt")
            print("At epoch {}, save model with lowest validation loss: {}".format(epoch, v_loss))
            
        v_losses.append(v_loss)
        v_avg_precisions.append(v_avg_precision)
        v_mean_avg_precisions.append(v_mean_avg_precision)
    return train_losses, train_avg_precisions, train_mean_avg_precisions, v_losses, v_avg_precisions, v_mean_avg_precisions



if __name__=='__main__':
    
    batch_size = 32
    maxnumepochs = 50
    learning_rate = 1e-3
    num_categories = 5
    num_imgs = 5
    transform = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])
    torch.manual_seed(0)
    init_folder = 'results1'
    
    if not os.path.exists(init_folder):
        os.makedirs(init_folder)
    else:
        shutil.rmtree(init_folder)
        os.makedirs(init_folder)

    dataset_train=PascalVOC('./data/VOCdevkit/VOC2012/', 'train', transform=transform)
    dataset_val=PascalVOC('./data/VOCdevkit/VOC2012/', 'trainval', transform=transform)
    dataset_test=PascalVOC('./data/VOCdevkit/VOC2012/', 'val', transform=transform)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    print('start')
    device=torch.device("cuda")
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 20)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0)
    criterion = nn.BCEWithLogitsLoss()
    
    train_losses, train_avg_precisions, train_mean_avg_precisions, v_losses, v_avg_precisions, v_mean_avg_precisions = train_modelcv(dataloader_cvtrain = train_loader, dataloader_cvVal = val_loader ,  model = model ,criterion = criterion, optimizer = optimizer, scheduler = None, num_epochs = maxnumepochs, batch_size = batch_size , device = device, init_folder=init_folder)

    print("Start Testing")
    
    model.load_state_dict(torch.load("model_parameter.pt"))
    test_avg_precisions, test_mean_avg_precisions, pred_scores, pred_labels, pred_imgs_path= evaluate_test(model = model, test_loader = test_loader, criterion = criterion, device = device)
    
    with open(init_folder + '/test_mean_avg_precisions.txt', 'w') as the_file:
        the_file.write(str(test_mean_avg_precisions*100))
        
    print('test mean average precision:  {:.2f}%'.format(test_mean_avg_precisions*100))
    tailacc, t_val = get_tail_acc(pred_scores.cpu().numpy(), pred_labels.cpu().numpy(), pred_imgs_path)
    
    with open(init_folder + '/tail_accuracy.txt', 'w') as the_file:
        the_file.write(str(tailacc) + " " + str(t_val))
        
    plot_loss_graph(maxnumepochs, train_losses, v_losses, init_folder)
    plot_accuracy_graph(maxnumepochs, train_mean_avg_precisions, v_mean_avg_precisions, init_folder)
    plot_tailacc_graph(t_val,tailacc, init_folder)
    
    predictions_sorted = []
    image_indexes_sorted = []
    images_path_sorted = []
    print('Total number of catageries available: ', len(dataset_test.list_image_sets()))
    for i in range(len(dataset_test.list_image_sets())):
        preds, img_idxs, imgs_path = get_rank_by_categories(pred_scores=pred_scores.cpu(), imgs_path=pred_imgs_path, col=i)
        images_path_sorted.append(imgs_path.tolist())
    image_indexes_sorted = imgs_path
    
    cat_indx_list= get_x_catagories_index(num_categories)
    print(cat_indx_list)
    
    get_top_x_imgs_categories(cat_indx_list, num_imgs, init_folder)
    get_last_x_imgs_categories(cat_indx_list, num_imgs, init_folder)