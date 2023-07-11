# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
import glob
import seaborn as sns
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.



def main():
    model = models.resnet18(pretrained='imagenet')

    # Resize the image to 224x224 px
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # Use the model object to select the desired layer
    layer = model._modules.get('avgpool')

    file_dir = "Analysis/wf4"
    file_lists = glob.glob(f"{file_dir}/object_*.jpg")

    list_features = [
        extract_feature_vector(Image.open(pathi), scaler, to_tensor, normalize, layer, model)
        for pathi in file_lists
    ]
    feature_matrix = np.array(list_features)
    scaled_feature = scale(feature_matrix)

    pca = PCA()
    res = pd.DataFrame(
        pca.fit_transform(X=scaled_feature)
    ).assign(fname=file_lists)
    res.columns = [str(i) for i in range(res.shape[1]-1)] + ["fname"]

    res_use = res.loc[:, ["fname", "0", "1"]]

    sns.scatterplot(data=res_use, x="0", y="1")
    plt.savefig("xxx.jpg")
    plt.close()


def extract_feature_vector(img, scaler, to_tensor, normalize, layer, model):
    # 2. Create a PyTorch Variable with the transformed image
    #Unsqueeze actually converts to a tensor by adding the number of images as another dimension.
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1, 512, 1, 1)

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # 6. Run the model on our transformed image
    model(t_img)

    # 7. Detach our copy function from the layer
    h.remove()

    # 8. Return the feature vector
    return my_embedding.squeeze().numpy()


def get_cosine_distance(path1, path2, scaler, to_tensor, normalize, layer, model):
    im1, im2 = Image.open(path1), Image.open(path2)
    image1 = extract_feature_vector(im1, scaler, to_tensor, normalize, layer, model).reshape(1, -1)
    image2 = extract_feature_vector(im2, scaler, to_tensor, normalize, layer, model).reshape(1, -1)
    return cosine_similarity(image1, image2)
