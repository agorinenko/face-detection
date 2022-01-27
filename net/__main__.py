import ssl

import numpy as np
import torch

from net import models, train
from net.criterions import MultiBoxLoss
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


if __name__ == "__main__":
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("mycreds.json")

    if gauth.credentials is None:
        gauth.LocalWebserverAuth()

    elif gauth.access_token_expired:
        gauth.Refresh()

    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile("mycreds.json")

    # ssl._create_default_https_context = ssl._create_unverified_context
    #
    # CLASSES = np.asarray([
    #     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    #     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    #     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    #     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    #     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    #     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    #     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    #     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    #     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    #     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    #     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    # ])
    # epochs = 1
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # print(f'DEVICE: {device}')
    #
    # model = models.SSD300(n_classes=len(CLASSES))
    # model = model.to(device)
    # criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    # train_loss_history, scores_history = train.train(model, epochs, criterion, device)
