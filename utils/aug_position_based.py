import json
import cv2
import numpy as np
import os
import csv
from PIL import Image
import shutil

CLASS_NAMES = ["Title", "xdata", "ydata", "xlabel", "ylabel", "xpoints", "ypoints"]
base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/recognizer/")
MAX_SIZE = 512

def load_labelbox_json():
    with open(base_path + "data_labels_v2.json") as json_file:
        label_data = json.load(json_file)

    print(len(label_data))
    labels = []
    for img_label in label_data:
        # read filename
        img_filename = img_label["External ID"]
        for class_name in CLASS_NAMES:
            left, top, width, height = 0, 0, 0, 0
            xpoints = [xpoint for xpoint in img_label["Label"]["objects"] if xpoint["title"] == class_name]
            for xpoint in xpoints:
                left = xpoint["bbox"]["left"]
                top = xpoint["bbox"]["top"]
                width = xpoint["bbox"]["width"]
                height = xpoint["bbox"]["height"]
                break

            labels.append({"image_names": img_filename, "cell_type": class_name, "xmin": left, "xmax": left + width, "ymin": top, "ymax": top + height})
    return labels

def scale_img(image, scale, x_shift, y_shift):
    new_size = (int(image.shape[0] * scale), int(image.shape[1] * scale))
    scaled_image = cv2.resize(image, new_size, interpolation = cv2.INTER_LANCZOS4)
    new_im = Image.new("RGB", (MAX_SIZE, MAX_SIZE), (255, 255, 255))
    scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)
    scaled_image = Image.fromarray(scaled_image)
    new_im.paste(scaled_image, (int((MAX_SIZE - new_size[0]) // 2 + x_shift), int((MAX_SIZE - new_size[1]) // 2 + y_shift)))
    new_im = cv2.cvtColor(np.asarray(new_im), cv2.COLOR_RGB2BGR)
    return new_im

def aug_images(labels):
    x_shift = np.linspace(-50, 50, 5)
    y_shift = np.linspace(-50, 200, 5)
    scales = np.linspace(0.8, 1.2, 5)
    print(len(scales))
    
    img_files = set([d['image_names'] for d in labels])
    shutil.rmtree(base_path + "imgs/aug_roi/x/", ignore_errors=True)
    os.mkdir(base_path + "imgs/aug_roi/x/")

    num = 0
    new_labels = []
    for img_file in img_files:
        img = cv2.imread(base_path + "imgs/resized/" + img_file)
        props = [i for i in labels if i["image_names"] == img_file]
        #print(props)
        for x_sh in x_shift:
            for y_sh in y_shift:
                for scale in scales:
                    print("image{:0>4d}.png".format(num), scale, x_sh, y_sh)
                    new_img = scale_img(img, scale, x_sh, y_sh)
                    cv2.imwrite(base_path + "imgs/aug_roi/x/image{:0>4d}.png".format(num), new_img)
                    for prop in props:
                        xmin = max(min(int(prop["xmin"])+ int(MAX_SIZE * (1 - scale) // 2 + x_sh), MAX_SIZE), 0)
                        xmax = max(min(int(prop["xmax"])+ int(MAX_SIZE * (1 - scale) // 2 + x_sh), MAX_SIZE), 0)
                        ymin = max(min(int(prop["ymin"])+ int(MAX_SIZE * (1 - scale) // 2 + y_sh), MAX_SIZE), 0)
                        ymax = max(min(int(prop["ymax"])+ int(MAX_SIZE * (1 - scale) // 2 + y_sh), MAX_SIZE), 0)
                        if (ymax == 0 and ymin == 0) or (xmax == 0 and xmin == 0):
                            continue
                        new_prop = {
                            "filepath": "image{:0>4d}.png".format(num), 
                            "x1": xmin, 
                            "y1": ymin, 
                            "x2": xmax, 
                            "y2": ymax,
                            "class_name": prop["cell_type"], 
                            #"oxmin": prop["xmin"],
                            #"oxmax": prop["xmax"],
                            #"oymin": prop["ymin"],
                            #"oymax": prop["ymax"],
                            #"xshift": x_sh,
                            #"yshift": y_sh,
                            #"scale": scale
                           } 
                        new_labels.append(new_prop)
                        #print(new_prop)
                    num += 1
        
    return new_labels

def generate_labeled_csv(filename):
    labels = load_labelbox_json()
    new_labels = aug_images(labels)
    with open(filename, 'w+', newline='') as csvfile:
        fieldnames = ['filepath', 'x1', 'y1', 'x2', 'y2', 'class_name']#,'oxmin', 'oxmax', 'oymin', 'oymax', 'xshift', 'yshift', 'scale']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in new_labels:
            writer.writerow(row)

if __name__ == "__main__":
    csv_filename = 'data/recognizer/label_data.csv'
    generate_labeled_csv(csv_filename)