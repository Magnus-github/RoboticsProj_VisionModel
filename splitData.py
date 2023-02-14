import os
import random
import shutil
import json


SRC_DIR = [
    "data/c0/",
    "data/c1/",
    "data/c2/",
    "data/c3/",
    "data/c4/",
    "data/c5/",
    "data/c6/",
    "data/c7/"
]

TRAIN_DIR = "data/training/"

VAL_DIR = "data/validation/"

if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)
if not os.path.exists(VAL_DIR):
    os.makedirs(VAL_DIR)

JSON_PATH = "data/annotations/merged.json"

TRAIN_JSON_PATH = "data/annotations/training.json"
VAL_JSON_PATH = "data/annotations/validation.json"


with open(JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)

train_images_lst = []
train_annotations_lst = []
val_images_lst = []
val_annotations_lst = []

train_info = {
        "contributor": "DD2419 students, 2023", 
        "url": "None", 
        "version": 1.0, 
        "description": "DD2419 objects dataset, training", 
        "year": 2023
}

val_info = {
        "contributor": "DD2419 students, 2023", 
        "url": "None", 
        "version": 1.0, 
        "description": "DD2419 objects dataset, validation", 
        "year": 2023
    }

# 80% training data and 20% validation, later test can be added
train_split = 0.8
val_split = 0.2

for dir in SRC_DIR:
    img_list = os.listdir(dir)
    abs_train_split = int(train_split * len(img_list))
    
    for i in range(abs_train_split):
        # randomly choose image
        a = random.choice(img_list)
        # copy to training folder and rename
        current_length = len(os.listdir(TRAIN_DIR))

        # check if image is in json
        image_index = next((index for (index, d) in enumerate(json_data["images"]) if d["file_name"] == "c" + str(SRC_DIR.index(dir)) + "/" + a), None)
        if image_index != None:

            shutil.copy(dir + a, TRAIN_DIR + "image" + str(current_length).zfill(4) + ".jpg")

            test = "c" + str(SRC_DIR.index(dir)) + "/" + a

            # update name in new json
            
            d_img = {
                "id": len(train_images_lst) + 1,
                "width": 1280,
                "height": 720,
                "file_name": "image" + str(current_length).zfill(4) + ".jpg"
            }
            train_images_lst.append(d_img)

            # find annotation using image_id and update it
            index = next((index for (index, d) in enumerate(json_data["annotations"]) if d["image_id"] == image_index+1), None)
            d_ann = json_data["annotations"][index]
            d_ann["id"] = len(train_annotations_lst)+1
            d_ann["image_id"] = d_img["id"]
            train_annotations_lst.append(d_ann)

        img_list.remove(a)

    # just copy rest of images in validation dir
    for img in img_list:
        current_length = len(os.listdir(VAL_DIR))

        image_index = next((index for (index, d) in enumerate(json_data["images"]) if d["file_name"] == "c" + str(SRC_DIR.index(dir)) + "/" + img), None)
        if image_index != None:

            shutil.copy(dir + img, VAL_DIR + "image" + str(current_length).zfill(4) + ".jpg")

            # find image id in json and update name in new json
            image_index = next((index for (index, d) in enumerate(json_data["images"]) if d["file_name"] == "c" + str(SRC_DIR.index(dir)) + "/" + img), None)
            d_img = {
                "id": len(val_images_lst) + 1,
                "width": 1280,
                "height": 720,
                "file_name": "image" + str(current_length).zfill(4) + ".jpg"
            }
            val_images_lst.append(d_img)

            # find annotation using image_id and update it
            index = next((index for (index, d) in enumerate(json_data["annotations"]) if d["image_id"] == image_index+1), None)
            d_ann = json_data["annotations"][index]
            d_ann["id"] = len(val_annotations_lst)+1
            d_ann["image_id"] = d_img["id"]
            val_annotations_lst.append(d_ann)

# write json files
train_json_data = {
        "info": train_info,
        "images": train_images_lst,
        "annotations": train_annotations_lst,
        "categories": json_data["categories"]
    }
with open(TRAIN_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(train_json_data, f, ensure_ascii=False, indent="\t")

val_json_data = {
        "info": val_info,
        "images": val_images_lst,
        "annotations": val_annotations_lst,
        "categories": json_data["categories"]
    }
with open(VAL_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(val_json_data, f, ensure_ascii=False, indent="\t")