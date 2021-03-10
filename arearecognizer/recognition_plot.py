import csv
import cv2
import pytesseract
import statistics
import numpy as np
from matplotlib import pyplot as plt
import configparser
from utils.func_timing import timer
from scipy.signal import find_peaks
from re import match as re_match


@timer
def calc_color_num(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel = img_hsv[:,:,0].ravel(), img_hsv[:,:,1].ravel()
    h_vals = []
    for i, val in enumerate(s_channel):
        if val > 10:
            h_vals.append(h_channel[i])

    count, bin = np.histogram(h_vals, bins=list(range(255)))
    count = count.astype(np.float)
    count[count == 0] = 1
    count_log = np.log(count)
    count_filter = filter_points(count_log, eps=0.95, stride=4)
    count_filter = np.array(filter_points(count_filter, eps=0.9, stride=2))

    peaks, _ = find_peaks(count_filter, distance=20)
    np.diff(peaks)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.figure()
    ax1.hist(h_vals, bin, log=True)
    ax2.plot(count_filter)
    ax2.plot(peaks, count_filter[peaks], "x")
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_title('HSV color histogram')
    ax2.set_title('HSV color peaks')
    plt.show()

    return max(len(peaks) + 1, 2)

def show_plots(images, cols=3):
    try:

        plt.figure(figsize=(10, 10))
        if len(images) == 1:
            f, ax = plt.subplots()
            ax.imshow(images[0])
        else:
            cols = min(cols, len(images))
            size_y = max(1, len(images) // cols)
            print('size', cols, size_y, len(images))
            fig, ax = plt.subplots(size_y, cols)#, figsize=(15, 15))
            for i in range(len(images)):
                if size_y == 1:
                    ax[i % cols].imshow(images[i])
                elif cols == 1:
                    ax[i].imshow(images[i])
                else:
                    ax[i // cols, i % cols].imshow(images[i])
        plt.show()
    except Exception  as ex:
        print(ex)


@timer
def pre_processing(image):
    """
    Converts input image to binary image
    :param image: image
    :return: thresholded image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    return threshold_img

@timer
def parse_text(threshold_img, config, isnum = False):
    """
    Feed input image to tesseract library to predict text.
    :param threshold_img: image
    return: meta-data dictionary
    """
    kernel = np.ones((2, 1), np.uint8)
    img = cv2.erode(threshold_img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)

    # configuring parameters for tesseract
    #/usr/share/tesseract-ocr/tessdata
    if isnum:
        tesseract_config = config['tesseract']['numbers_params']
    else:
        tesseract_config = config['tesseract']['text_params']
    # now feeding image to tesseract
    details = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                        config=tesseract_config, lang='eng')
    return format_text(details, threshold_img.shape)

@timer
def draw_boxes(image, details, threshold_point = 10):
    """
    This function draws boxes on text area detected by Tesseract. It also writes resulted image to local disk.
    :param image: image
    :param details: dictionary
    :param threshold_point: integer
    :return: None
    """
    total_boxes = len(details['text'])
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for sequence_number in range(total_boxes):
        if int(details['conf'][sequence_number]) > threshold_point:
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                            details['width'][sequence_number], details['height'][sequence_number])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return image

@timer
def format_text(details, img_shape):
    """
    This function arrange resulted text into proper format (join words to phrases).
    :param details: dictionary
    :return: list
    """
    vpadding = int(img_shape[1] / 90)
    hpadding = int(img_shape[0] / 90 )
    parse_text_lst = []
    parse_pos_lst = []
    word_list = []
    pos_list = []
    last_word = ''
    for i, word in enumerate(details['text']):
        if word != '':
            word_list.append(word)
            pos_list.append((details['left'][i], details['top'][i], details['width'][i], details['height'][i]))
            last_word = word
        if ((last_word != '' and word == '') or (word == details['text'][-1])) and len(word_list) > 0:
            parse_text_lst.append(word_list)

            left = min([i[0] for i in pos_list])
            top = min([i[1] for i in pos_list])

            if len(pos_list) > 1:
                width_deviation = statistics.stdev([i[2] for i in pos_list])
                height_deviation = statistics.stdev([i[3] for i in pos_list])
                # if text is horizontal
                if width_deviation > height_deviation:
                    width = sum([i[2] for i in pos_list]) + len(pos_list) * vpadding
                    height = max([i[3] for i in pos_list])
                # if text is vertical
                else:
                    width = max([i[2] for i in pos_list])
                    height = sum([i[3] for i in pos_list])  + len(pos_list) * hpadding
            else:
                width = pos_list[0][2]
                height = pos_list[0][3]

            parse_pos_lst.append((left, top, width, height))
            word_list = []
            pos_list = []
    return parse_text_lst, parse_pos_lst

@timer
def write_text(formatted_text, filename):
    """
    Writes arranged text into a file.
    :param formatted_text: list
    :return: None
    """
    with open(filename, 'w', newline="") as file:
        csv.writer(file, delimiter=" ").writerows(formatted_text)

@timer
def draw_arranged_boxes(image, text, boxes):
    """
    This function draws bounding boxes over arranged text.
    :param img: image
    :param text: string
    :param boxes: list
    :return: None
    """
    img = image.copy()
    for i in range(len(boxes)):
        img = cv2.rectangle(
            img,
            (boxes[i][0], boxes[i][1]),
            (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]),
            (0, 0, 255),
            1
        )
        cv2.putText(img, ' '.join(text[i]), (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return img

@timer
def filter_points(points, alpha = 0.8, eps = 10, stride=1):
    new_points = [points[0]]
    for i in range(1, len(points)):
        p = points[i] + alpha * (new_points[-1] - points[i])
        if (len(new_points) > 2) and (np.mean(new_points[min(i, 0 - stride): -1]) - new_points[-2] < eps):
            new_points.append(p)
        else:
            new_points.append(points[i])
    return new_points

@timer
def detect_lines(img, orig_img):
    orig_image = orig_img.copy()
    image = img.copy()
    image = cv2.GaussianBlur(image,(5,5),0)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(image, 150, 200, apertureSize=3)

    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    points = np.array([[0, 0]])
    for i, c in enumerate(contours):
        rect = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if rect[2] * rect[3] < (img.shape[0] * img.shape[1] / 100) or rect[2] * rect[3] / area < 2:
            continue
        x, y, w, h = rect
        con = np.array(c[:,0])
        for i, val in enumerate(con):
            if len(points) == 1 or (not val[0] in points[:, 0]):
                vals = [i[1] for i in con if i[0] == val[0]]
                points = np.vstack((points, [val[0], int(np.mean(vals))]))

        cv2.rectangle(orig_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.drawContours(orig_image, [c], -1, (255, 0, 0), 1)

    points = points[1:]
    if (len(points) > 1):
        points = points[points[:, 0].argsort()]
    return edges, orig_image, points

@timer
def reduce_color_kmeans(img, colors = 8):
    """
    Reduce number of used colors on image with k-means clustering
    :param img: source image
    :param colors: target number of colors
    :return: reduced image
    """
    print(f'number of clusters: {colors}')
    # img_data = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img_data[:,:,2].fill(255)# = img_data[:,:,1] * 3
    # img_data = cv2.cvtColor(img_data, cv2.COLOR_HSV2BGR)
    # img_data = img_data / 255.0
    img_data = img / 255.0
    img_data = img_data.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    img_data = img_data.astype(np.float32)
    compactness, labels, centers = cv2.kmeans(img_data, colors, None, criteria, 10, flags)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter([i[0] for i in centers], [[i[1] for i in centers]], [[i[2] for i in centers]], depthshade=True)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    white_label = np.argmax(np.sum(centers, axis=1))
    new_colors = (centers[labels].reshape(img.shape) * 255).astype('uint8')
    masked_images = []

    for c in range(colors):
        c_labels = np.copy(labels)
        c_labels[c_labels != c] = white_label
        masked_images.append((centers[c_labels].reshape(img.shape) * 255).astype('uint8'))
    return new_colors, masked_images

def is_number_regex(s):
    """ Returns True if string is a number. """
    if re_match("^\-?\d*\.?\d*$", s) is None:
        return s.isdigit()
    return True

@timer
def recognize_plot(img_orig, config, need_display_plt=False):
    cluster_num = calc_color_num(img_orig)
    img_reduced, masked_imgs = reduce_color_kmeans(img_orig, cluster_num)
    img_reduceed_threshoulded = pre_processing(img_reduced)
    #img_edges, img_lines, points = detect_lines(img_reduceed_threshoulded, img_reduced)
    #show_images([[img_reduced, img_reduceed_threshoulded], [img_edges, img_lines]], "reduced images")
    masked_edges = []
    masked_lines = []
    result = []
    for masked_img in masked_imgs:
        img_masked_thresh = pre_processing(masked_img)
        img_edges, img_lines, points = detect_lines(img_masked_thresh, masked_img)
        masked_edges.append(img_edges)
        masked_lines.append(img_lines)
        if len(points) > 1:
            x_vals = points[1:, 0]
            y_vals = np.max(points[1:, 1]) - points[1:, 1]
            y2_vals = filter_points(y_vals)
            result.append([x_vals, y_vals, y2_vals])
            if need_display_plt:
                plt.plot(x_vals, y_vals, 'b-', x_vals, y2_vals, 'r-')
                plt.show()

    if need_display_plt:
        show_plots(masked_lines, 2)
        show_plots(masked_edges, 2)
    return result

def show_texts(img_orig, config):
    # calling pre_processing function to perform pre-processing on input image.
    img_threshoulded = pre_processing(img_orig)
    # calling parse_text function to get text from image by Tesseract.
    arranged_num, arranged_num_boxes = parse_text(img_threshoulded, config)
    print(arranged_num, arranged_num_boxes)
    # calling draw_boxes function which will draw dox around text area.
    img_arranged_boxes = draw_arranged_boxes(img_orig, arranged_num, arranged_num_boxes)
    # calling write_text function which will write arranged text into file
    write_text(arranged_num, '../data/resulted_text.txt')
    show_plots([img_arranged_boxes], 1)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("settings.ini")

    img_orig = cv2.imread(config['storage']['filename'])
    recognize_plot(img_orig, config, need_display_plt=True)
    show_texts(img_orig, config)
