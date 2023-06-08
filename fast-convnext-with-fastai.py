#!/usr/bin/env python
# coding: utf-8


# competition = 'paddy-disease-classification'
path = "xxxxx"

from fastai.vision.all import *

set_seed(42)
import timm
import os


# train images
train_path = f'{path}/train_images'
train_files = get_image_files(train_path)

# test images
test_path = f'{path}/test_images'
test_files = get_image_files(test_path).sorted()

# sample submission
sample_submission = pd.read_csv(f'{path}/sample_submission.csv')
num_epoch = 100
# train labels
train_df = pd.read_csv(f'{path}/train.csv')
train_df.head()
batch_size = 64
lr = 0.005 * batch_size / 32.

# make output folder
model_name = 'convnext_tiny'
out_dir = "Analysis/paddy_doctor/fastai_convnext"
os.makedirs(out_dir, exist_ok=True)

# ### Dataloaders for fastai training

splitter = TrainTestSplitter(test_size=0.1, random_state=42, stratify=train_df.label.values)

# dblock = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),
#     get_items=get_image_files,
#     get_y=parent_label,
#     splitter=RandomSplitter(0.1, seed=42),
#     item_tfms=Resize(480, method='squish'),
#     batch_tfms=aug_transforms(size=224, min_scale=0.75)
# )
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    get_y=parent_label,
    splitter=splitter,
    item_tfms=Resize(480, method='squish'),
    batch_tfms=aug_transforms(size=224, min_scale=0.75)
)

dls = dblock.dataloaders(train_path,
                         bs=batch_size)

# 2. By using the high level `ImageDataLoaders`


dls.show_batch()

# ### Create a learner and train

# In[9]:


# model_ft = timm.create_model(model_name='convnext_tiny_in22k', pretrained=True)
learn = vision_learner(dls, model_name, metrics=error_rate).to_fp16()

# And that's it, 16 epochs to get the best baseline for the price

# In[10]:


learn.fine_tune(num_epoch, lr)

# ### Predictions and Test Time Augmentation

# Lets compare the error rate -on the validation set- that are obtained with the normal prediction function and with the predictions we can get applying a technique called Test Time Augmentation (TTA). As you'll see, TTA is easy with fastai.

# In[11]:


# Get predictions on validation set
probs, target = learn.get_preds(dl=dls.valid)
error_rate(probs, target)

# In[12]:


# Get TTA predictions on validation set
probs, target = learn.tta(dl=dls.valid)
error_rate(probs, target)

# In[13]:


qqq = error_rate(probs, target)
print(f"tta error {float(qqq):.4f}")

# So you can see a boost with TTA.

# ### Predictions on test set

# In[14]:


# TTA predictions from test images
probs, _ = learn.tta(dl=dls.test_dl(test_files))

# In[15]:


# get the index with the greater probability
preds = probs.argmax(dim=1)

# In[17]:


dls.vocab[preds]

# ### Submission

# In[18]:


sample_submission.label = dls.vocab[preds]
sample_submission.to_csv(f'{out_dir}/submission.csv', index=False)

# save model
learn.save(f'{out_dir}/model')

# ### Conclusions