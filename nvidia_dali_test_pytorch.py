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

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import plotly.express as px
from plotly.subplots import make_subplots

import warnings
import glob
from sklearn import preprocessing
from sklearn.metrics import f1_score
import ttach as tta
import timm

__print__ = print
warnings.filterwarnings('ignore')


def print(string):
    os.system(f'echo \"{string}\"')
    __print__(string)


tta_transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.9, 1, 1.1]),
    ]
)

##################
# You can modify input directory.
DIR_input = "data/xxxx"

SEED = 43
N_FOLDS = 10
N_EPOCHS = 100
# N_EPOCHS = 1 # for test
BATCH_SIZE = 40  # for b4
# BATCH_SIZE = 75 # for b2
# SIZE = 260
SIZE = 380
# lr = 5 * 10 ** (-4) * BATCH_SIZE / 64
lr = 5 * 10 ** (-5) * BATCH_SIZE / 64
# lr = 5 * 10 ** (-3) * BATCH_SIZE / 64 # for b4
# lr = 5 * 10 ** (-4)


test_img_path = f"{DIR_input}/test_images"
train_img_path = f"{DIR_input}/train_images"
# You can also modify output directory
out_dir = "Analysis/xxxx"
os.makedirs(out_dir, exist_ok=True)

transforms_tta = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.9, 1, 1.1]),
    ]
)


class DALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, df, im_size, num_class, dir_input=DIR_input,
                 random_shuffle=True, train=True):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id)
        self.img_ids = df

        self.indexes = df.index.values
        # self.labels = label_binarize(df.labels.values, classes=np.arange(num_class))
        self.labels = df.labels.values
        self.dir_input = dir_input

        breeds = list(self.labels)
        self.breed2idx = {b: i for i, b in enumerate(breeds)}

        img_list = pd.DataFrame({'data': df.img_path, 'label': self.labels})
        img_list.to_csv('dali.txt', header=False, index=False, sep=' ')

        self.input = fn.readers.file(file_root='.', file_list='dali.txt', random_shuffle=random_shuffle)
        # self.decode = fn.decoders.image_crop(self.input, device = "mixed", output_type = types.DALIImageType.RGB)
        # self.decode = ops.ImageDecoderRandomCrop(device = "mixed", output_type = types.DALIImageType.RGB)
        # self.resize = fn.random_resized_crop(device = "gpu", size=(224, 224))
        self.transpose = ops.Transpose(device='gpu', perm=[2, 0, 1])
        # self.cast = fn.cast(device='gpu', dtype=types.DALIDataType.FLOAT)
        self.im_size = im_size
        self.train = train

    def define_graph(self):
        images, labels = self.input
        images = fn.decoders.image_crop(images, device="mixed", output_type=types.DALIImageType.RGB)
        if self.train:
            images = fn.random_resized_crop(images, device="gpu", size=(self.im_size, self.im_size))
        else:
            images = fn.resize(images, device="gpu", size=(self.im_size, self.im_size))
        images = fn.cast(images, device='gpu', dtype=types.DALIDataType.FLOAT16)
        output = self.transpose(images)
        # output = images
        return output, labels


def DALIDataLoader(batch_size, df, dir_input, im_size, num_class, random_shuffle=True, train=True):
    num_gpus = 1
    pipes = [
        DALIPipeline(batch_size=batch_size, num_threads=2, device_id=device_id, df=df,
                     random_shuffle=random_shuffle, dir_input=dir_input, num_class=num_class, im_size=im_size,
                     train=train) for device_id in
        range(num_gpus)]

    pipes[0].build()
    epoch_dic = pipes[0].epoch_size()
    epoch_size = list(epoch_dic.values())[0]

    dali_iter = DALIGenericIterator(pipelines=pipes, output_map=['data', 'label'],
                                    size=epoch_size, reader_name=None,
                                    auto_reset=True, fill_last_batch=True, dynamic_shape=False,
                                    last_batch_padded=True)
    return dali_iter


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
                   df_train, df_valid, transforms_train, transforms_valid, num_class=12):
    train_fold_results = []
    device = torch.device("cuda:0")
    best_val_score = -10 ** 5

    for epoch in range(N_EPOCHS):
        os.system(f'echo \"  Epoch {epoch}\"')
        dataloader_train = DALIDataLoader(batch_size=BATCH_SIZE, df=df_train, dir_input=DIR_input, random_shuffle=True,
                                          im_size=SIZE, num_class=num_class, train=True)

        model.train()
        tr_loss = 0

        for step, batch in enumerate(dataloader_train):
            # print(f"training step {step + 1}")
            images = transforms_train(batch[0]["data"])
            # images = transforms_train(batch[0]["data"]).half()
            # images = transforms_train(batch[0]["data"])
            # labels = batch[0]["label"].unsqueeze(-1)
            labels = torch.from_numpy(label_binarize(batch[0]["label"], classes=np.arange(num_class))).unsqueeze(-1)

            # images = transforms_train(images)
            labels = labels.to(device, dtype=torch.float).half()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels.squeeze(-1))
            loss.backward()

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

        # to prepare validation
        num_train = len(dataloader_train)
        del dataloader_train
        torch.cuda.empty_cache()
        dataloader_valid = DALIDataLoader(batch_size=BATCH_SIZE, df=df_valid, dir_input=DIR_input, random_shuffle=False,
                                          num_class=num_class, im_size=SIZE, train=False)

        # validate
        model.eval()
        val_loss = 0
        val_preds = None
        val_labels = None

        for step, batch in enumerate(dataloader_valid):
            # print(f"validation step {step + 1}")
            images = transforms_valid(batch[0]["data"])
            labels = torch.from_numpy(label_binarize(batch[0]["label"], classes=np.arange(num_class))).unsqueeze(-1)

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1).detach().cpu()
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1).detach().cpu()),
                                       dim=0)

            # images = transforms_valid(images.to(device, dtype=torch.float))
            labels = labels.to(device, dtype=torch.float).half()

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels.squeeze(-1))
                val_loss += loss.item()

                preds = torch.nn.LogSoftmax(dim=1)(outputs).exp().cpu()

                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)

            torch.cuda.empty_cache()

        num_valid = len(dataloader_valid)
        del dataloader_valid
        torch.cuda.empty_cache()

        # accuracy_score = roc_auc_score(val_labels, val_preds, average="macro", multi_class="ovo")
        f1_score_pred = f1_score(val_labels.numpy().argmax(1), val_preds.numpy().argmax(1), average='weighted')
        simple_accuracy = (val_labels.numpy().argmax(1) == val_preds.numpy().argmax(1)).sum() / val_labels.shape[0]

        result_dict = {
            "fold": i_fold,
            "epoch": epoch,
            "train_loss": tr_loss / num_train,
            "valid_loss": val_loss / num_valid,
            # "valid_score": accuracy_score,
            "f1_score_pred": f1_score_pred,
            "accuracy": simple_accuracy
        }

        train_fold_results.append(result_dict)
        print("prediction score")
        print(result_dict)

    return val_preds, train_fold_results


def test_step(model, transforms_test, submission_df, num_class):
    model.eval()
    test_preds = None
    dataloader_test = DALIDataLoader(batch_size=min(BATCH_SIZE, submission_df.shape[0]), df=submission_df,
                                     dir_input=DIR_input,
                                     random_shuffle=False, im_size=SIZE, num_class=num_class, train=False)

    for step, batch in enumerate(dataloader_test):

        images = transforms_test(batch[0]["data"])
        labels = batch[0]["label"].unsqueeze(-1)

        with torch.no_grad():
            outputs = model(images.float())

            if test_preds is None:
                test_preds = outputs.data.cpu()
            else:
                test_preds = torch.cat((test_preds, outputs.data.cpu()), dim=0)

    del dataloader_test
    torch.cuda.empty_cache()

    return test_preds


def test_step_tta(model, transforms_test, submission_df, num_class):
    model.eval()
    test_preds = None
    dataloader_test = DALIDataLoader(batch_size=min(BATCH_SIZE, submission_df.shape[0]), df=submission_df,
                                     dir_input=DIR_input,
                                     random_shuffle=False, im_size=SIZE // 4, num_class=num_class, train=False)
    tta_model = tta.ClassificationTTAWrapper(model, transforms_tta)

    for step, batch in enumerate(dataloader_test):

        images = transforms_test(batch[0]["data"])
        labels = batch[0]["label"].unsqueeze(-1)

        with torch.no_grad():
            # outputs = model(images.float())
            outputs = tta_model(images.float())

            if test_preds is None:
                test_preds = outputs.data.cpu()
            else:
                test_preds = torch.cat((test_preds, outputs.data.cpu()), dim=0)

    del dataloader_test
    del images
    torch.cuda.empty_cache()

    return test_preds


def define_model(num_class):
    # model_ft = models.vgg16(pretrained=True)
    # num_ftrs = model_ft.classifier[-1].in_features
    # for mi, m in enumerate(model_ft.classifier):
    #     # print(mi)
    #     for p in m.parameters():
    #         p.required_grad = True
    # model_ft.classifier[-1] = nn.Linear(num_ftrs, num_class, bias=True)

    # model_ft = PlantModel(
    #     num_classes=num_class
    # )

    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, num_class)
    # model_ft = models.efficientnet_b0(pretrained=True)
    # model_ft = models.efficientnet_b2(pretrained=True)

    # effnet4
    # model_ft = models.efficientnet_b4(pretrained=True)
    # num_ftrs = model_ft.classifier[1].in_features
    # model_ft.classifier[1] = nn.Linear(num_ftrs, num_class)
    # model_ft.classifier[0] = nn.Dropout(p=0.3, inplace=True)

    # model_ft = models.inception_v3(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, num_class)
    model_ft = models.convnext_tiny(pretrained=True)
    num_ftrs = model_ft.classifier[2].in_features
    model_ft.classifier[2] = nn.Linear(num_ftrs, num_class)

    return model_ft


def oversampling_data():
    """
    increase the size of dataset with random oversampling
    :return:
    """

    path_dali = "dali.txt"
    path_dali_out = "dali_revised.txt"
    seed = 777

    df1 = pd.read_table(path_dali, header=None, sep=" ")
    np.random.seed(seed)
    classes = df1[1].unique()

    max_count = df1.groupby([1]).count()[
                (df1.groupby([1]).count()==df1.groupby([1]).count().max())[0].values
    ][0].iloc[0]

    sampled_list = []

    for i in classes:
        dfi = df1.loc[df1[1]==i]

        sample_num = max_count - dfi.shape[0]
        df_list = []
        if sample_num>0:
            sampled_paths = np.random.choice(dfi[0].values, size=sample_num)
            df_add = pd.DataFrame({"0": sampled_paths, "1": i}).rename(
                {"0": 0, "1": 1} ,axis=1
            )
            df_list.append(df_add)
        df_list.append(dfi)
        df_use = pd.concat(df_list)
        sampled_list.append(df_use)

    df_sampled = pd.concat(sampled_list)

    df_sampled.to_csv(path_dali_out, sep=" ", header=False, index=False)


def main2():
    mean_std = torch.Tensor([0.5, 0.5, 0.5]) * 255

    ## korniaでtransform
    transforms_train = nn.Sequential(
        # K.RandomResizedCrop(size=(SIZE, SIZE), p=1.0),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
        K.RandomRotation(degrees=45, p=0.8),
        K.RandomMotionBlur(3, 35., 0.5),
        K.RandomSharpness(p=0.5),
        K.Normalize(mean=mean_std, std=mean_std, p=1.0),
    ).half()

    transforms_valid = nn.Sequential(
        # K.Resize(size=(SIZE, SIZE), p=1.0),
        K.Normalize(mean=mean_std, std=mean_std, p=1.0),
    ).half()

    ## サイズ制限のためのtransform
    # transforms_size = A.Compose([
    #     A.RandomResizedCrop(height=SIZE,
    #                         width=SIZE, p=1.0),
    #     ToTensorV2(p=1.0)])

    #####Preparing submission file####
    submission_df = pd.read_csv(f"{DIR_input}/sample_submission.csv").assign(
        img_path=lambda x: test_img_path + "/" + x.image_id
    )
    # submission_df.iloc[:, 1:] = 0
    submission_df["labels"] = 0

    ######Preparing data set########
    train_df = pd.read_csv(f"{DIR_input}/train.csv").assign(
        img_path=lambda x: train_img_path + "/" + x.label + "/" + x.image_id
    )
    num_class = train_df["label"].unique().shape[0]

    # For debugging.
    # train_df = train_df.sample(n=100)
    # train_df.reset_index(drop=True, inplace=True)
    le = preprocessing.LabelEncoder()
    le.fit(train_df.iloc[:, 1].values)

    train_labels = le.transform(train_df.label.values)

    # Need for the StratifiedKFold split
    train_y = train_labels
    train_df["labels"] = train_y

    # For training model..
    folds = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    )
    oof_preds = np.zeros((train_df.shape[0]))

    submissions = None
    train_results = []

    torch.cuda.empty_cache()

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):
        print(f"Fold {i_fold + 1}/{N_FOLDS}")

        valid = train_df.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)

        train = train_df.iloc[train_idx]
        train.reset_index(drop=True, inplace=True)
        device = torch.device("cuda:0")

        # dataloader_train = DALIDataLoader(batch_size=BATCH_SIZE, df=train, dir_input=DIR_input, random_shuffle=True,
        #                                   im_size=SIZE, dir_test_train="train_images")

        model = define_model(num_class=num_class).to(device)

        criterion = DenseCrossEntropy()
        # criterion = torch.nn.CrossEntropyLoss()
        plist = [
            {"params": model.parameters(),
             "lr": lr}
        ]
        optimizer = optim.Adam(plist, lr=lr)

        val_preds, train_fold_results = train_one_fold(
            i_fold=i_fold, model=model, criterion=criterion, optimizer=optimizer,
            df_train=train, df_valid=valid,
            transforms_train=transforms_train, transforms_valid=transforms_valid,
            num_class=num_class
        )
        oof_preds[valid_idx] = val_preds.argmax(1).numpy()[0:len(valid_idx)]

        train_results = train_results + train_fold_results

        # test_preds = test_step_tta(model, transforms_test=transforms_valid,
        #                            submission_df=submission_df, num_class=num_class)
        test_preds = test_step(model, transforms_test=transforms_valid,
                               submission_df=submission_df, num_class=num_class)

        # Save predictions per fold
        pred_label_pre = torch.nn.LogSoftmax(dim=1)(test_preds).exp().cpu().numpy().argmax(1)
        pred_label = le.inverse_transform(pred_label_pre)
        submission_df["label"] = pred_label[0:submission_df.shape[0]]
        submission_df.drop(["img_path", "labels"], axis=1).to_csv(f'{out_dir}/submission_fold_{i_fold}.csv',
                                                                  index=False)

        # logits avg
        if submissions is None:
            submissions = test_preds / N_FOLDS
        else:
            submissions += test_preds / N_FOLDS

        torch.cuda.empty_cache()

    print("5-Folds CV score: {:.4f}".format(roc_auc_score(train_labels, oof_preds, average='macro', multi_class="ovo")))


if __name__ == '__main__':
    # main()
    main2()

    # epoch 51くらいでスコアがだだ下がりしてしまう。。
