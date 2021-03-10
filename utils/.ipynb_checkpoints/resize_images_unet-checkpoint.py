import os
import cv2
import numpy as np


def resize_image(image, size = 512, inter = cv2.INTER_AREA):
    orig_shape = max(image.shape[0], image.shape[1])
    new_img = np.ones((orig_shape, orig_shape, image.shape[2]), np.uint8) * 255
    new_img[:image.shape[0], :image.shape[1], :3] = image
    return cv2.resize(new_img, (size, size), interpolation=inter)

def create_output_masked_tensor(img_filename, class_names, img_size = (512, 512), file_dir ="", is_train=False):
    if is_train:
        base_path = ""
        output_tensor = np.ones((img_size[0], img_size[1], len(class_names)), np.uint8) * 255
    else:
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/recognizer/imgs/")
        output_tensor = np.ones((img_size[0], img_size[1], 1, len(class_names)), np.uint8) * 255
    for i, output_channel in enumerate(class_names):
        #print(base_path + file_dir + output_channel + "/" + img_filename)
        channel = cv2.imread(base_path + file_dir + output_channel + "/" + img_filename)
        if is_train:
            output_tensor[:, :, i] = channel[:, :, 0]
        else:
            output_tensor[:, :, 0, i] = channel[:, :, 0]

    return output_tensor

def resize_original():
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/recognizer/imgs/")
    orig_path = base_path + "original/"
    target_path = base_path + "resized/"
    for img_path in os.listdir(orig_path):
        img = cv2.imread(orig_path + img_path)
        cv2.imwrite(target_path + img_path, resize_image(img, 512, cv2.INTER_AREA))

if __name__ == "__main__":
    resize_original()
    # create_output_masked_tensor("test02.png", ["Title", "xdata", "ydata", "xlabel", "ylabel", "xpoints", "ypoints"])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
