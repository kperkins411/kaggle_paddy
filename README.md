# kaggle_paddy competition
https://www.kaggle.com/competitions/paddy-disease-classification

2 solutions, 1 using pytorch and one using fastAI, both share the same mutihead model, both use timm vision models.

## Pytorch Solution:
 Multihead model: forward returns (label,variety) tuple  
  DataSet: __getitem__ returns image,lbl,variety  
  loss function: __call__ takes a preds tuple and correct_labels,correct_varieties  
  TIMM model extended to provide 2 outputs.  
 Learning rate selected via optuna optimizer  

## FastAI port
 Uses above model and loss  
 Learning rate selected via fastai learning rate finder  


## Observations:
Fastai learning rate finder is much faster and easier to use than optuna


