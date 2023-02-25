import os
import random
import shutil
import json
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.datasets import CocoDetection

import utils
from detector import Detector



def main():
    detector = Detector().to(device="cuda")

    dataset = CocoDetection(
        root="./data/training",
        annFile="./data/annotations/training.json",
        transforms=detector.input_transform,
    )
    val_dataset = CocoDetection(
        root="./data/validation",
        annFile="./data/annotations/validation.json",
        transforms=detector.input_transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    i=0
    for train_img, train_target in dataloader:
        figure, ax = plt.subplots(1)
        bbs = detector.decode_output(train_target)

        plt.imshow(train_img[0,:,:,:].cpu().permute(1,2,0))
        plt.imshow(
            train_target[0,4,:,:],
            interpolation="nearest",
            extent=(0,1280,720,0),
            alpha=0.7,
        )
        utils.add_bounding_boxes(ax, bbs[0],showClass=True)
        i+=1
        if i > 10:
            break
    
    print("Showing images:")
    plt.show()


if __name__ == "__main__":
    main()

    # train_images = []
    # val_images = []
    # train_dir = "./data/training"
    # val_dir = "./data/validation"
    # if not os.path.exists(train_dir):
    #     print("Training directory '%s' does not exist!" %train_dir)
    #     return
    # if not os.path.exists(val_dir):
    #     print("Training directory '%s' does not exist!" %val_dir)
    #     return
    # Nt = len(os.listdir(train_dir))
    # Nv = len(os.listdir(val_dir))
    # train_anns = torch.zeros((Nt,13,detector.out_cells_y,detector.out_cells_x))
    # val_anns = torch.zeros((Nv,13,detector.out_cells_y,detector.out_cells_x))
    # for file_name in os.listdir(train_dir):
    #     if file_name.endswith(".jpg"):
    #         file_path = os.path.join(train_dir, file_name)
    #         train_image = Image.open(file_path)
    #         torch_image, target = detector.input_transform(train_image, [])
    #         train_images.append(torch_image)
    # for file_name in os.listdir(val_dir):
    #     if file_name.endswith(".jpg"):
    #         file_path = os.path.join(val_dir, file_name)
    #         val_image = Image.open(file_path)
    #         torch_image, _ = detector.input_transform(val_image, [])
    #         val_images.append(torch_image)
    
    # if train_images and val_images:
    #     train_images = torch.stack(train_images)
    #     train_images = train_images.to(device="gpu")
    #     val_images = torch.stack(val_images)
    #     val_images = val_images.to(device="gpu")

    # bbs = detector.decode_output(train_targets)
    