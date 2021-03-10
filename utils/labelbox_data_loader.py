import json
import cv2
import numpy as np
import os


def draw_label(image, img_label, label):
    new_img = image.copy()
    xpoints = [xpoint for xpoint in img_label["Label"]["objects"] if xpoint["title"] == label]
    for xpoint in xpoints:
        left = xpoint["bbox"]["left"]
        top = xpoint["bbox"]["top"]
        width = xpoint["bbox"]["width"]
        height = xpoint["bbox"]["height"]
        new_img[top: top + height, left: left + width] = 255
    return new_img

def load_labelbox_json():
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/recognizer/")
    with open(base_path + "data_labels_v2.json") as json_file:
        label_data = json.load(json_file)

    print(len(label_data))
    for img_label in label_data:
        # read filename
        img_filename = img_label["External ID"]
        img = cv2.imread(base_path + "imgs/resized/" + img_filename)
        # create blank BW image
        image = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        image[:] = 0
        for class_name in ["Title", "xdata", "ydata", "xlabel", "ylabel", "xpoints", "ypoints"]:
            cv2.imwrite(base_path + "imgs/" + class_name + "/" + img_filename, draw_label(image, img_label, class_name))

if __name__ == "__main__":
    load_labelbox_json()