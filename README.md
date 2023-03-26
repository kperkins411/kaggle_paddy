# kaggle_paddy competition
https://www.kaggle.com/competitions/paddy-disease-classification

2 solutions, 1 using pytorch and one using fastAI, both share the same mutihead model, both use timm vision models.

 Multihead attention with custom DataSet,  error and loss function.  TIMM model extended to provide 2 outputs.


A pure pytorch solution
multihead model
custom dataset that returns image and 2 labels
custom error and loss function
Learning rate selected via optuna optimizer


And a port of above solution to FastAI
Learning rate selected via fastai learning rate finder


Observations:

Fastai learning rate finder is much faster and easier to use than optuna


