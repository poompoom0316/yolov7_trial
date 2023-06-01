import os
import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, models, transforms

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize
import kornia.augmentation as K

import jpeg4py as jpeg

import plotly.express as px
from plotly.subplots import make_subplots

import warnings

warnings.filterwarnings('ignore')

from sklearn import preprocessing

__print__ = print


def print(string):
    os.system(f'echo \"{string}\"')
    __print__(string)


##################
DIR_input = "data"

SEED = 43
N_FOLDS = 5
N_EPOCHS = 25
BATCH_SIZE = 64
SIZE = 512
lr = 5 * 10 ** (-5) * BATCH_SIZE / 64

label_columns = []


class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, DIR_INPUT=DIR_input):
        self.df = df
        self.transforms = transforms
        self.dir_input = DIR_INPUT
        self.persistent_workers = True
        self.indexes = df.index.values
        self.labels = df.labels.values

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = f"{self.dir_input}/train_images/{self.df.loc[idx, 'image']}"

        # image = cv2.imread(image_src, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)
        image = jpeg.JPEG(image_src).decode()
        labels = torch.tensor(
            [self.labels[self.indexes == idx][0]]
        ).unsqueeze(-1)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image)

        return image, labels


class PlantModel(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = torchvision.models.resnet18(pretrained=True)
        in_featurs = self.backbone.fc.in_features
        self.logit = nn.Linear(in_featurs, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.25, self.training)

        x = self.logit(x)

        return x


class DenseCrossEntropy(nn.Module):
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        logprobbs = F.log_softmax(logits, dim=-1)
        loss = -labels * logprobbs
        loss = loss.sum(-1)

        # print(f"loss: {loss}")

        return loss.mean()


def train_one_fold(i_fold, model, criterion, optimizer,
                   dataloader_train, dataloader_valid, transforms_train, transforms_valid):
    train_fold_results = []
    device = torch.device("cuda:0")

    for epoch in range(N_EPOCHS):
        os.system(f'echo \"  Epoch {epoch}\"')

        model.train()
        tr_loss = 0

        for step, batch in enumerate(dataloader_train):
            print(f"training step {step + 1}")
            images = batch[0]
            labels = batch[1]

            images = transforms_train(images.to(device, dtype=torch.float))
            labels = labels.to(device, dtype=torch.float)

            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(-1))
            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

        # validate
        model.eval()
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, batch in enumerate(dataloader_valid):
            print(f"validation step {step + 1}")
            images = batch[0]
            labels = batch[1]

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)),
                                       dim=0)

            images = transforms_valid(images.to(device, dtype=torch.float))
            labels = labels.to(device, dtype=torch.float)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels.squeeze(-1))
                val_loss += loss.item()

                preds = torch.softmax(outputs, dim=1).data.cpu()

                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)

        binarized_label = label_binarize(val_labels, classes=range(12))
        accuracy_score = roc_auc_score(binarized_label, val_preds, average="macro", multi_class="ovo")
        print(f"accuracy score: {accuracy_score}")

        train_fold_results.append(
            {
                "fold": i_fold,
                "epoch": epoch,
                "train_loss": tr_loss / len(dataloader_train),
                "valid_loss": val_loss / len(dataloader_valid),
                "valid_score": accuracy_score
            }
        )

    return val_preds, train_fold_results


def define_model(num_class):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_class)
    return model_ft


def main():
    mean_std = torch.Tensor([0.5, 0.5, 0.5]) * 255

    ## korniaでtransform
    transforms_train = nn.Sequential(
        K.RandomResizedCrop(size=(SIZE, SIZE), p=1.0),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
        K.RandomRotation(degrees=45, p=0.8),
        K.RandomMotionBlur(3, 35., 0.5),
        K.RandomSharpness(p=0.5),
        K.Normalize(mean=mean_std, std=mean_std, p=1.0),
    )

    transforms_valid = nn.Sequential(
        K.Resize(size=(SIZE, SIZE), p=1.0),
        K.Normalize(mean=mean_std, std=mean_std, p=1.0),
    )

    ## サイズ制限のためのtransform
    transforms_size = A.Compose([
        A.RandomResizedCrop(height=SIZE,
                            width=SIZE, p=1.0),
        ToTensorV2(p=1.0)])

    #####Preparing submission file####
    submission_df = pd.read_csv(f"{DIR_input}/sample_submission.csv")
    submission_df.iloc[:, 1:] = 0

    ######Preparing data set########
    device0 = torch.device("cuda:0")
    dataset_test = PlantDataset(df=submission_df, transforms=transforms_size)
    dataloader_test = DataLoader(
        dataset=dataset_test, batch_size=BATCH_SIZE, num_workers=1,
        shuffle=False, prefetch_factor=2)

    train_df = pd.read_csv(f"{DIR_input}/train.csv")

    # For debugging.
    # train_df = train_df.sample(n=100)
    # train_df.reset_index(drop=True, inplace=True)
    le = preprocessing.LabelEncoder()
    le.fit(train_df.iloc[:, 1].values)

    train_labels = le.transform(train_df.labels.values)

    # Need for the StratifiedKFold split
    train_y = train_labels
    train_df["labels"] = train_y

    train_df.head()

    # For training model..
    folds = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    oof_preds = np.zeros((train_df.shape[0]))

    model = PlantModel(
        num_classes=train_df.iloc[:, 1].unique().shape[0]
    )

    submissions = None
    train_results = []

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):
        print(f"Fold {i_fold + 1}/{N_FOLDS}")

        valid = train_df.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)

        train = train_df.iloc[train_idx]
        train.reset_index(drop=True, inplace=True)
        device = torch.device("cuda:0")

        dataset_train = PlantDataset(df=train, transforms=transforms_size)
        dataset_valid = PlantDataset(df=valid, transforms=transforms_size)

        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                      num_workers=1, shuffle=True, persistent_workers=True, prefetch_factor=2)
        dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE,
                                      num_workers=1, shuffle=False, persistent_workers=True, prefetch_factor=2)

        model = define_model(num_class=12)
        model.to(device)

        criterion = DenseCrossEntropy()
        plist = [
            {"params": model.parameters(),
             "lr": lr}
        ]
        optimizer = optim.Adam(plist, lr=lr)

        val_preds, train_fold_results = train_one_fold(
            i_fold=i_fold, model=model, criterion=criterion, optimizer=optimizer,
            dataloader_train=dataloader_train, dataloader_valid=dataloader_valid,
            transforms_train=transforms_train, transforms_valid=transforms_valid
        )
        oof_preds[valid_idx, :] = val_preds.numpy()

        train_results = train_results + train_fold_results

        model.eval()
        test_preds = None

        for step, batch in enumerate(dataloader_test):

            images = batch[0]
            images = images.to(device, dtype=torch.float)

            with torch.no_grad():
                outputs = model(images)

                if test_preds is None:
                    test_preds = outputs.data.cpu()
                else:
                    test_preds = torch.cat((test_preds, outputs.data.cpu()), dim=0)

        # Save predictions per fold
        submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_preds, dim=1)
        submission_df.to_csv('submission_fold_{}.csv'.format(i_fold), index=False)

        # logits avg
        if submissions is None:
            submissions = test_preds / N_FOLDS
        else:
            submissions += test_preds / N_FOLDS

    print("5-Folds CV score: {:.4f}".format(roc_auc_score(train_labels, oof_preds, average='macro')))


if __name__ == '__main__':
    main()
