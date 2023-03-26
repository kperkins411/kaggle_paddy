import torch.nn as nn
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torchvision.transforms as transforms
import torch
import pandas as pd
import torch.optim as optim

#--------------------------
#configuration
#-------------------------- 
# from https://www.kaggle.com/code/hinepo/transfer-learning-with-timm-models-and-pytorch
class PROJECT_CFG:
    train_path='./data/train_images'   #get train and validation Datasets from here
    test_path='./data/test_images'     #get test set from here
    csv_path='./data/train.csv'
    
    ### split train and validation sets
    split_fraction = 0.2

    ##dataloader
    drop_last=False
    
    ### model
    model_name = 'resnet26d'#'convnext_small_in22k' # ## 'resnet50' # 'resnet34', 'resnet200d', 'efficientnet_b1_pruned', 'efficientnetv2_m', efficientnet_b7 ...  

    num_classes_in_output_layer=512
    #get a subset of data to work on(start with True until all experiments done
    #and want to train on entire dataset
    subsample_data=False
    BATCH_SIZE= 8 if subsample_data else 32
    N_EPOCHS = 10
    print_freq = 4 

    momentum=0.9
    
    ### set only one to True
    save_best_loss = True
    save_best_accuracy = False

    ### optimizer
    # optimizer = 'adam'
    # optimizer = 'adamw'
    optimizer = 'rmsprop'  
    # LEARNING_RATE=0.00001
    LEARNING_RATE_MIN=0.003
    LEARNING_RATE_MAX=0.01

    random_seed = 42


#--------------------------
#learner statistics
#-------------------------- 
class stats():
    '''
    tracks losses and error rates for labels and varieties
    determines if current loss is best
    prints out loss info (update user on model progress)
    '''
    def __init__(self,kind='train'):
        self.reset()
        self.kind=kind
        self.best_loss_labels = None
        self.best_loss_varieties = None
        self.best_err_rate_labels =None
        self.best_err_rate_varieties =None
        self.PROJECT_CFG=PROJECT_CFG()
            
    def reset(self):
        self.running_loss_labels = 0.0
        self.running_loss_varieties = 0.0
        self.running_err_rate_labels =0.0
        self.running_err_rate_varieties =0.0
        
    def add(self,loss_labels,loss_varieties,err_rate_labels,err_rate_varieties):
        self.running_loss_labels+=loss_labels
        self.running_loss_varieties+=loss_varieties
        self.running_err_rate_labels+=err_rate_labels
        self.running_err_rate_varieties+=err_rate_varieties
        
    def is_best_loss(self,btch_num):
        '''
        call at end of every epoch to see if we should save model
        '''
        ret=False #assumme worst
        loss_lbls, loss_varieties,err_rate_labels,err_rate_varieties=self.calc_curr_loss(btch_num)
        
        if self.PROJECT_CFG.save_best_accuracy:
            if (self.best_err_rate_labels is None) or(self.best_err_rate_labels>err_rate_labels):
                self.best_err_rate_labels=err_rate_labels
                ret= True
        else:
            if (self.best_loss_labels is None) or(self.best_loss_labels>loss_lbls):
                self.best_loss_labels=loss_lbls
                ret= True
                
        self.reset()
        return ret
 
    def calc_curr_loss(self,btch_num):
        den=btch_num+1
        return self.running_loss_labels/den,self.running_loss_varieties/den,self.running_err_rate_labels/den,self.running_err_rate_varieties/den
        
    def show(self,btch_num):
        '''
        prints out losses
        '''
        if(btch_num%self.PROJECT_CFG.print_freq == 0):
            loss_lbls, loss_varieties,err_rate_labels,err_rate_varieties=self.calc_curr_loss(btch_num)
            print(f'{self.kind}:label accuracy={1-err_rate_labels:.2f},   labelloss={loss_lbls:.2f},  variety accuracy={1-err_rate_varieties:.2f} variety loss={loss_varieties:.2f}', end='\r', flush=True)
#--------------------------
#Multi Head Model (2 output params)
#--------------------------  
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR  

class DiseaseAndTypeClassifier(nn.Module):
    def __init__(self,tmodel):
        '''
        tmodel: pretrained model
        ex:
        model_name='resnet26d'
        tmodel=timm.create_model(model_name, pretrained=True)
        m1=DiseaseAndTypeClassifier(tmodel)
        
        '''
        super().__init__()
        self.m = tmodel
        self.l1=nn.Linear(in_features=self.m.get_classifier().out_features, out_features=10, bias=False)  #rice type
        self.l2=nn.Linear(in_features=self.m.get_classifier().out_features, out_features=10, bias=False)  #disease
        
    def forward(self,x):       
        x=F.relu(self.m(x)) 
        label=self.l1(x)  #disease type
        variety=self.l2(x)  #variety
        
        #return as tuple
        return (label,variety)

def set_fine_tune(m1, train_these_layers):
    '''
    set all layers requires_grad=False in model unless 
    layer name is in train these layers
    ex.
    set_fine_tune(m1,['m.fc.weight','m.fc.bias','l1.weight','l2.weight'])   
    '''
    set_all_layer_grad(m1,False)
    
    for name, param in m1.named_parameters():
        for ln in train_these_layers:
            if ln in name:
                param.requires_grad=True
                # print(name)
                continue

def set_all_layer_grad(m1,grad=True):
    '''
    set all layers equires_grad=grad
    layer name is in train these layers
    ex.
    set_fine_tune(m1,['m.fc.weight','m.fc.bias','l1.weight','l2.weight'])   
    '''
    for name, param in m1.named_parameters():
        param.requires_grad=grad
  
#--------------------------
#Simple Learner class
#--------------------------  
from torch.optim.lr_scheduler import OneCycleLR       
class Learner():
    def __init__(self,m,scheduler,optimizer):
        '''
        m: model to train with
        scheduler
        optimizer
        '''
        #the “safer” approach would be to move the model to the device first and create the optimizer afterwards.
        self.m=m        
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.m.to(self.device)
        
        self.criterion=DiseaseAndTypeClassifierLoss()
        
        #learning rate scheduler
        self.scheduler=scheduler
        
        #optimizer
        self.optimizer=optimizer
        
        self.trn_stats=stats()
        self.val_stats=stats('valid')
        self.lrs=[]  #used for verifing 1 cycle performence
        self.architecture=self.m.m.default_cfg['architecture']
        
    def learn(self,trn_dl,val_dl, num_epochs): 
        self.lrs=[]
              
        # loop over the dataset num_epochs times
        for epoch in range(num_epochs):           
            self._trn_epoch(trn_dl)           
            self._val_epoch(val_dl,epoch)

    def _trn_epoch(self,dl):                                                                                                                          
        self.trn_stats.reset()
        self.m.train()
        
        for i, data in (enumerate(dl)):
            
             # get the inputs, labels is tuple(label, variety)
            imgs,lbls,varietys = data[0].to(self.device),data[1].to(self.device),data[2].to(self.device)
            
            # zero the parameter gradients
            self.optimizer.zero_grad()
 
            # forward + backward + optimize
            preds = self.m(imgs)             
            loss_labels , loss_varieties = self.criterion(preds, lbls,varietys)
            
            #see https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
            loss_labels.backward(retain_graph=True)
            loss_varieties.backward()

            #save info                                                                                                             
            self.trn_stats.add(loss_labels.item(),loss_varieties.item(),error_rate(preds[0],lbls),error_rate(preds[1],varietys))
  
            #adjust weights
            self.optimizer.step()
 
            #adjust learning rate
            self.scheduler.step()
            self.lrs.append( self.scheduler.get_last_lr())
            
            self.trn_stats.show(i)  
        print()
          
    def _val_epoch(self,dl,epoch):                                                                                                                          
        self.val_stats.reset()
        self.m.eval()
        with torch.no_grad():
            for i, data in (enumerate(dl)):

                 # get the inputs, labels is tuple(label, variety)
                imgs,lbls,varietys = data[0].to(self.device),data[1].to(self.device),data[2].to(self.device)

                # forward + backward + optimize
                preds = self.m(imgs)             
                loss_labels , loss_varieties = self.criterion(preds, lbls,varietys)
                
                #save loss
                self.val_stats.add(loss_labels.item(),loss_varieties.item(),error_rate(preds[0],lbls),error_rate(preds[1],varietys))

                self.val_stats.show(i)

            if(self.val_stats.is_best_loss(i)):
                print(f'\nEpoch {epoch}found and saving better model')
                 # save the model's weights and biases 
                torch.save(self.m.state_dict(), f"./BEST_{self.architecture}.pth")               
            print()


def get_Learner(model_name, min_lr,max_lr,num_epochs,trn_dl,num_output_classes,momentum,load_best_model=False):
    '''
    gets a timm model and wraps in a  DiseaseAndTypeClassifier class, use this if using pur pytorch
    min_lr: minimum learning rate for OneCycleLR
    max_lr: maximum learning rate for OneCycleLR
    num_epochs: how many epochs to train for
    trn_dl: TRAIN dataloader-used by OneCycleLR to calculate correct number of steps
    cfg:
    '''
    #create the timm model
    tmodel=timm.create_model(model_name, pretrained=True, num_classes=num_output_classes,global_pool='catavgmax') 

    #and pass it to DiseaseAndTypeClassifier
    m1=DiseaseAndTypeClassifier(tmodel)
    
    if(load_best_model):
        m1.load_state_dict(torch.load(f"./BEST_{m1.m.default_cfg['architecture']}.pth"))  
     
    #create optimizer
    optimizer=optim.SGD(m1.parameters(),min_lr, momentum=momentum)

    #create a learning rate scheduler
    scheduler = OneCycleLR(optimizer, 
                max_lr = max_lr, # Upper learning rate boundaries in the cycle for each parameter group
                steps_per_epoch=int(len(trn_dl)),epochs=num_epochs,
                anneal_strategy = 'cos') # Specifies the annealing strategy

    #create the learner that will train the model
    return Learner(m1,scheduler,optimizer)

#--------------------------
#metrics
#--------------------------          
def acc(preds,lbls):
    '''
    preds:[nxn] tensor
    lbls:[1xn] tensor
    sums number of lbl positions that are equivalent to max pred
    then divides by total number of samples
    returns: float between 0 and 1
    '''
    return ((torch.argmax(preds,dim=1)==lbls).sum()/len(lbls)).item()
def error_rate(preds,lbls):
    return(1-acc(preds,lbls))

def get_accuracy(m,dl, verbose=True):
    '''
    returns the models accuracy from both labl and variety
    
    m1.trained model
    dl: DataLoader with data of interest (train or validation)    
    return: acc_labl,acc_var
    '''
    #where is model (send data to evaluate there)
    device=next(m.parameters()).device.type  #return cuda if on cuda
    
    m.eval()
    cum_acc_lbl=0
    cum_acc_varietie=0
    for i,data in enumerate(dl):
        img,lbl,variety = data[0].to(device),data[1].to(device),data[2].to(device)
        pred_lbl,pred_varieties = m(img)
        cum_acc_lbl+=acc(pred_lbl,lbl)
        cum_acc_varietie+=acc(pred_varieties,variety)
    acc_labl=(cum_acc_lbl*100)/(i+1)
    acc_var=(cum_acc_varietie*100)/(i+1)
    if verbose: print(f'Accuracy: Label prediction={ acc_labl:.2f}%,  Variety prediction={ acc_var:.2f}')
    return acc_labl,acc_var

#--------------------------
#loss function stuff
#--------------------------
class DiseaseAndTypeClassifierLoss(nn.Module):
    '''
    calculates the cross entropy loss for both labels and varieties
    and returns them as a tuple
    '''
    def __init__(self):
        super(DiseaseAndTypeClassifierLoss, self).__init__()

    def forward(self,preds, correct_labels,correct_varieties):
        #split out the preds
        pred_labels,pred_varieties=preds
        
        #using crossentropy loss
        criterion = nn.CrossEntropyLoss()
        
        loss_labels= criterion(pred_labels, correct_labels)
        loss_varieties=criterion(pred_varieties, correct_varieties)
    
        return loss_labels , loss_varieties
    
    def __call__(self,preds, correct_labels,correct_varieties):
        #this is here strictly for FASTAI learner, it expects a 1 number loss (not a tuple)
        return self.forward(preds, correct_labels,correct_varieties)
        
#--------------------------
#dataset stuff
#--------------------------
def get_fls(pth, ext=['.png','.jpg']):
    '''
    pth: recursively gather all image files in pth
    return: list of files (with absolute paths)
    '''
    pth=os.path.abspath(pth)       #absolute path
    ext=[ext.lower() for ext in ext]  #make sure lowercase
    res=[]
    for root,dirs,fles in os.walk(pth):
        for fle in fles:
            if os.path.splitext(fle)[1].lower() in ext:
                res.append(os.path.join(root,fle))
    return res

class MultiTaskDatasetTest(Dataset):
    '''
    Use for test dataset, all files are in img_dir, there is NO other information associated with each file               
    '''
    def __init__(self,img_dir,*,transforms=None, target_transform=None):
        '''
        img_dir:where files are (assummes no subfolders)
        transforms: Use the same transforms that you train and validate with
        '''
        self.files=get_fls(img_dir)  #build a list of all the absolute path files
        self.transforms = transforms
        self.target_transform = target_transform  
     
    def __len__(self): return len(self.files)
        
    def __getitem__(self,idx):
        image=read_image(str(self.files[idx]))   #get the image    
        if self.transforms:
            image = self.transforms(image)      
        return image

class mapper():
    '''
    forward and reverse map labels and varieties
    '''
    def __init__(self,df):
        #df: dataframe from './data/train.csv'     
        # create a dict to map label strings to tensors
        self.i_to_label=dict(enumerate(df.label.unique()))
        self.label_to_i={v:i for i,v in self.i_to_label.items()}
        #create another to map variety strings to tensors
        self.i_to_variety=dict(enumerate(df.variety.unique()))
        self.variety_to_i={v:i for i,v in self.i_to_variety.items()}

class MultiTaskDatasetTrain(Dataset): 
    '''
    Use for train and validation datasets, use when you know both the label and variety               
    '''
    def __init__(self,img_dir,mpr,*,df=None,transforms=None, target_transform=None):
        '''
        df: contains info to build train OR validation datasets
        img_dir: where images are ex "./data/train_images"
        mpr:forward and reverse map labels and varieties
        transforms: list of transforms to apply      
        '''
        super().__init__()
        self.image_ids=df.loc[:,'image_id'].tolist()
        self.labels=df.loc[:,'label'].tolist()
        self.varietys=df.loc[:,'variety'].tolist()
        self.files=[]
        for image_id,label in zip(self.image_ids, self.labels):
            self.files.append(os.path.join(img_dir,label,image_id))
        assert len(self.files)==len(self.labels) and len(self.files)==len(self.varietys), f"files,labels and variety must be same length"

        #now convert the labels and varieties to numbers
        self.labels=list(map(mpr.label_to_i.get,self.labels))
        self.varietys=list(map(mpr.variety_to_i.get,self.varietys))
 
        self.transforms = transforms
        self.target_transform = target_transform       
        # self.size = len(self.files)

    def __len__(self): return len(self.files)

    def __getitem__(self,idx):
        #dealing with the image
        image=read_image(str(self.files[idx]))
        
        lbl=self.labels[idx]
        variety=self.varietys[idx]
        
        if self.transforms:
            image = self.transforms(image)
        # img = PIL.Image.open(self.paths[idx]).convert('RGB')
        # img = Image(pil2tensor(img, dtype=np.float32).div_(255))
        # img = img.apply_tfms(self.transform, size = self.size)
        # img = self.norm(img.data)
        
        return image,lbl,variety
    
    # def display(self,idx):
    #     image,lbl,variety=self[idx]
    #     print(f'{self.i_to_label[lbl]}')
    #     print(f'{self.i_to_variety[variety]}')
        # from fastai.vision.all import *
        # img=PILImage.create(files[0])
        # print(img.size)
        # img.to_thumb(128)


def get_transforms(cfg):  
    '''
    get the transforms needed by the datasets to configure the images
    
    cfg: comes from querying timm model 
    ex. cfg=timm.data.resolve_data_config({}, model=CFG.model_name, verbose=True)
        cfg  will be
        {'input_size': (3, 224, 224),
         'interpolation': 'bicubic',
         'mean': (0.485, 0.456, 0.406),
         'std': (0.229, 0.224, 0.225),
         'crop_pct': 0.875}        
    '''

    image_size=cfg['input_size'][-2:] #get just the height and width (not the channels)
    mean=cfg['mean'] #mean and std come from the model
    std=cfg['std']

    train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float),
            # transforms.ToTensor(), #not needed if already a tensor
            transforms.Normalize(mean=mean,std=std)     #mean and std come from the model
        ])
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=mean,std=std)
    ])
    
    return train_transforms, val_transforms

#--------------------------
#submission
#--------------------------
def generate_submission(m, df):
    '''
    m the model
    '''
    from torch.utils.data import DataLoader
    
    CFG=PROJECT_CFG()
    mpr=mapper(df)
    
    import timm
    cfg=timm.data.resolve_data_config({}, model=CFG.model_name, verbose=True)
    _, val_transforms=get_transforms(cfg)
    
    #generate sorted list of files
    test_dataset=MultiTaskDatasetTest(CFG.test_path, transforms=val_transforms) 
    test_dataset.files.sort()

    #generate a non_shuffle dataloader
    tst_dl=DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=1,drop_last=False)

    #where is the model
    device = next(m.parameters()).device.type

    #get model predictions
    labels=[]
    for imgs in tst_dl:
        imgs = imgs.to(device)

        # forward + backward + optimize
        pred_lbls,_ = m(imgs)

        #get the max index
        pred_lbls=torch.argmax(pred_lbls,dim=1).tolist()
        labels=labels+pred_lbls   

    #convert indexes to labels
    new_labels=[mpr.i_to_label[i] for i in labels]
    # get a list of files
    files=[fle.split('/')[-1] for fle in test_dataset.files]   

    with open('./submission.csv','w') as fle:
        fle.write('image_id,label')
        for i in range(len(files)):
            fle.write('\n'+files[i]+','+new_labels[i])