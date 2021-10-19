# Recognition approach
## The training uses arcface_torch (insightface git repo) sundirectory. Use im2rec.py file to convert your dataset to rec format for training.

### This subfolder contaions implementation of unusual approach to facial expressions estimation.
### The gist of the approach is to train a model using arcface loss as face recognition. Simply put, we have few identities (expression classes) and huge number of the identity samples.
### The model learns how to distinguish expressions and learns to yield expression coresponding logits. ***It's important that additionaly the model has to be provided with reference images representing each expression very vividly so that reference images logits have the lowest Eucledian distance to its target expression. For this reason, the model perormance depends a lot on reference images. Take into account that threre can be many reference images referencing to one expression***
### The folder content is as follows:
+ Expression references - contains images used as references when validating model performance
+ ResNet18 - contains training logs of resnet18
+ ResNet34 - contains training logs of resnet34
