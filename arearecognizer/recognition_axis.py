import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.models import load_model
from matplotlib import pyplot as plt
import cv2
import math
import arearecognizer.recognition_plot as pt
import pandas as pd
import utils.image_resizer as resizer
import configparser

# number of image channels (RGB or grayscale). So it should have values 3 or 1
n_channels = 3

# target classes for recognition (image order is important)
CLASS_NAMES = ["Title", "xdata", "ydata", "xlabel", "ylabel", "xpoints", "ypoints"]


def filter_points(points, bias = 0):
    '''
    Detect points that are located on the same distance between each other.
    Used by axis points recognition
    @param points: source points
    @param bias:
    @return: filtered points
    '''
    distances = []
    # get all distances between points
    if len(points) > 1:
        for i, dist in enumerate(points):
            d = abs(points[i][0] - points[i + 1][0])
            # points that are located closely to each other not used
            if d > 512 / 40:
                distances.append((points[i][0], d))
            # last point also used in all cases
            if i >= len(points) - 2:
                distances.append((points[i + 1][0], d))
                break
    # print(distances)
    # make distance frequency dictionary
    fr_dist = []
    for i in distances:
        found = False
        for j, fr in enumerate(fr_dist):
            if abs(fr[0] - i[1]) < 3:
                fr_dist[j][1] += 1
                found = True
                break
        if not found:
            fr_dist.append([i[1], 1])

    # select the most frequent distance to list of [distance, count] list
    # print('fr_dist', fr_dist)
    max_pos, imax_pos = 0, 0
    for i, a in enumerate(fr_dist):
        if a[1] >= max_pos:
            max_pos = a[1]
            imax_pos = i
    # print('dist', fr_dist[imax_pos])

    # filter distances that are similar to most frequent distance
    new_dist = []
    for i, dist in enumerate(distances):
        #print(point[1], fr_dist[imax_pos][0])
        if abs(dist[1] - fr_dist[imax_pos][0]) < 5:
            new_dist.append(dist)
    if len(new_dist) > 0:
        new_dist.append((new_dist[-1][0] + fr_dist[imax_pos][0], fr_dist[imax_pos][0]))

    #print('filter points before', bias, is_x, new_dist)
    for i, p in enumerate(new_dist):
        new_dist[i] = (new_dist[i][0] + bias, new_dist[i][1])
        #new_dist[i] = (new_dist[i][0] + (bias if not is_x else 0), new_dist[i][1] + (bias if is_x else 0))
    #print('filter points after', new_dist)

    return new_dist

def detect_area(src_img, preds, channel_num=0):
    '''
    Detect image area that is located over specific mask
    @param src_img: source image
    @param preds: recognized areas with channels
    @param channel_num: channel number (CLASS_NAMES)
    @return: target masked image and coordinates of the bounding rectangle
    '''
    img = src_img.copy()
    sel_points = preds[0, channel_num, :, :]

    # find largest contour among all areas
    contours, hier = cv2.findContours(sel_points, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    max_area = 0
    x, y, w, h = 0, 0, 0, 0
    for c in contours:
        xc, yc, wc, hc = cv2.boundingRect(c)
        if (wc * hc) > max_area and wc > 6 and hc > 6:
            x, y, w, h = xc, yc, wc, hc
            max_area = wc * hc

    if (w * h) > 100:
        #print(f'area {channel_num}: ', x, y, w, h)
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                if not (x < i < x + w and y < j < y + h):
                    img[j, i] = [0, 0, 0]
                else:
                    img[j, i] = img[j, i] * 255
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img, x, y, w, h

def detect_text(src_img, preds, positions, channel_num = 0):
    '''
    Detect texts that are located over mask
    @param src_img: source image
    @param preds: recognized areas with channels
    @param positions: texts positions
    @param channel_num: channel number (CLASS_NAMES)
    @return: identificators of texts that belongs to masked image
    '''
    img, x, y, w, h = detect_area(src_img, preds, channel_num)
    sel_idx = []
    thresh = 10
    for i, pos in enumerate(positions):
        center = pos[0] + pos[2] / 2
        if center > (x - thresh) and center < (x + w + thresh) \
                and pos[1] > (y - thresh) and (pos[1] + pos[3] < (y + h + thresh)):
            sel_idx.append(i)
    return sel_idx

def detect_points(src_img, preds, eps=1.0, is_x_axis=True, need_display=False):
    '''
    Detect points for plot scaling
    @param src_img: source image
    @param preds: recognized areas with channels (CLASS_NAMES)
    @param is_x_axis: is it X axis or Y
    @param need_display: should generate visualization?
    @return: list of points. Each point is a tuple of (position meaningful, relative position of another axis - not used)
    '''
    # алгоритм: 1. получить координаты рисок, 2. получить численные значения по оси x, y 3. определить для каждой риски значения из 2 пункта 4.
    # channel for x points or y points accordingly
    img, x, y, w, h = detect_area(src_img, preds, channel_num=5 if is_x_axis else 6)
    # find axis line
    dst = cv2.Canny(img, 50, 200, None, 3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    x_vals, y_vals = [], []

    if lines is not None:
        for i in range(0, len(lines)):
            rho, theta = lines[i][0][0], lines[i][0][1]
            a, b = math.cos(theta), math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 - 1000 * b), int(y0 + 1000 * a))
            pt2 = (int(x0 + 1000 * b), int(y0 - 1000 * a))
            x_vals.append(x0)
            y_vals.append(y0)
            # cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    # among all lines, find a line in the middle of the area of interest
    #print(x_vals)
    points = []
    if len(x_vals) > 0:
        x_idx = (np.abs(x_vals - np.mean(x_vals))).argmin()
        y_idx = (np.abs(y_vals - np.mean(y_vals))).argmin()
        x_line = int(x_vals[x_idx])
        y_line = int(y_vals[y_idx])
        thresh = 7
        # sliding window for finding crosses
        if is_x_axis:
            x_points = [(np.mean(img[y_line - thresh: y_line + thresh, i] / 255)) for i in range(x, x + w)]
            points = [(i, pos) for i, pos in enumerate(x_points) if pos < np.mean(x_points) / eps]
            points = filter_points(points, x)
            #print('x points ', points)
            if need_display:
                for i in points:
                    #cv2.line(img, (int(x + i[0]), 0), (int(x + i[0]), 512), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.line(img, (int(i[0]), 0), (int(i[0]), 512), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(img, (0, y_line), (512, y_line), (255, 0, 0), 1, cv2.LINE_AA)
        else:
            y_points = [(np.mean(img[i, x_line - thresh: x_line + thresh] / 255)) for i in range(y, y + h)]
            points = [(i, pos) for i, pos in enumerate(y_points) if pos < np.mean(y_points) / eps]
            points = filter_points(points, y)
            if need_display:
                for i in points:
                    #cv2.line(img, (0, int(y + i[0])), (512, int(y + i[0])), (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.line(img, (0, int(i[0])), (512, int(i[0])), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(img, (x_line, 0), (x_line, 512), (255, 0, 0), 1, cv2.LINE_AA)

    if need_display:
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.plot()
    print('points len', len(points))
    return points

def calc_axis_scale(points, texts, text_positions, is_x_axis=True):
    '''
    Detect scale between real plot and image coordinates
    @param points: points on image with small cross
    @param texts: axis texts with cross values
    @param text_positions: positions of that texts
    @param is_x_axis: is it X axis or not
    @return: a, b values for function y = a * x + b, where y - point coordinates, x - axis values from text
    '''
    #print('text', texts, text_positions)
    result = []
    for idx_p, point in enumerate(points):
        imin, min = -1, 512
        for idx_pos, pos in enumerate(text_positions):
            center_x = pos[0] + pos[2] / 2
            center_y = pos[1] + pos[3] / 2
            if is_x_axis:
                dist = abs((center_x - point[0]))
            else:
                #print('dist', center_y, point[0])
                dist = abs((center_y - point[0]))
            if dist < min:
                min = dist
                imin = idx_pos
        if imin > -1 and pt.is_number_regex(texts[imin]) and min < 5:
            result.append([point, min, texts[imin]])
    #print('result', result)
    pairs = []
    for i, val_i in enumerate(result):
        for j, val_j in enumerate(result):
            if i == j or \
                    len(val_i[2]) == 0 or \
                    len(val_j[2]) == 0 or \
                    (float(val_i[2]) - float(val_j[2])) == 0:
                continue
            #print('vals', val_i, val_j)
            a = (float(val_i[0][0]) - float(val_j[0][0])) / (float(val_i[2]) - float(val_j[2]))
            b = float(val_i[0][0]) - float(val_i[2]) * a
            true_num = 0
            for k, val_k in enumerate(result):
                if (len(val_k[2]) == 0): continue
                val = a * float(val_k[2]) + b
                if abs(val - float(val_k[0][0])) < float(val_k[0][0] * 0.05):
                    true_num += 1
            pairs.append([a, b ,true_num])
    #print('pairs', pairs)
    if len(pairs) == 0:
        return 0, 0
    res = sorted(pairs, key=lambda x: x[2], reverse=True)[0]
    #print('res', res)
    return res[0], res[1]#sorted(result, key=lambda x:x[1], reverse=False)

def filter_num(txt):
    '''
    Filtering of a text. It should contains only numbers and dots.
    If string starts with 0, point should be added
    @param txt: source text
    @return: filtered number as a string
    '''
    res = ""
    for i in range(len(txt)):
        if txt[i] in "0123456789.":
            res += txt[i]
    if res.startswith("0") and not res.startswith("0."):
        res = "0." + res[1:len(res)]
    return res

def recognize_axis(img, config):
    '''
    Axis recognition
    @param img: source image
    @return: tuple: title_vals - title of a plot, xlabel - x label, ylabel - y label, x_scale - calc_axis_scale result for X, y_scale - calc_axis_scale result for Y
    '''
    img_mask = img.copy()
    model = load_model(config['model']['model_load_path'], compile=False)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # optimizer = tf.keras.optimizers.Adam(lr=1e-6)
    optimizer = tf.keras.optimizers.SGD(lr=1e-7)
    metrics = [tf.keras.metrics.categorical_accuracy]
    model.compile(optimizer, loss, metrics)
    print(model.summary())

    img_mask = resizer.resize_image(img_mask)
    if n_channels == 1:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        img_mask = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    #print(img_mask.shape)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.asarray(img_mask, dtype=np.int))
    plt.show()
    img_mask = np.asarray(img_mask, dtype=np.float32) // 255
    img_mask = np.asarray(img_mask, dtype=np.uint8)
    # X = np.zeros((1, 512, 512, n_channels))
    X = np.zeros((1, n_channels, 512, 512))
    for i in range(n_channels):
        X[0, i, :, :] = img_mask[:, :, i]

    preds_train = model.predict(X, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_test_upsampled = []
    # print(f'preds len {len(preds_train)}')
    for i in range(len(preds_train)):
        preds_test_upsampled.append(np.squeeze(preds_train[i]))

    #print(preds_train_t[0, 0, :, :].shape)

    img_threshoulded = pt.pre_processing(img)
    texts, texts_pos = pt.parse_text(img_threshoulded, config)
    scale = img_mask.shape[1] / img.shape[1]
    #print(img_mask.shape[1], img.shape[1], scale)
    for i in range(len(texts_pos)):
        texts_pos[i] = (int(texts_pos[i][0] * scale),
                        int(texts_pos[i][1] * scale),
                        int(texts_pos[i][2] * scale),
                        int(texts_pos[i][3] * scale))
    #print(texts, texts_pos)
    title_idx = detect_text(img_mask, preds_train_t, texts_pos, 0)
    x_vals_idx = detect_text(img_mask, preds_train_t, texts_pos, 1)
    y_vals_idx = detect_text(img_mask, preds_train_t, texts_pos, 2)
    x_label_idx = detect_text(img_mask, preds_train_t, texts_pos, 3)
    y_label_idx = detect_text(img_mask, preds_train_t, texts_pos, 4)
    title_vals = [texts[i] for i in title_idx]
    x_vals = [filter_num(texts[i][0]) for i in x_vals_idx]
    y_vals = [filter_num(texts[i][0]) for i in y_vals_idx]
    xlabel = [texts[i] for i in x_label_idx]
    ylabel = [texts[i] for i in y_label_idx]

    print('title', title_vals)
    print('xvals', x_vals)
    print('yvals', y_vals)
    print('xlabel', xlabel)
    print('ylabel', ylabel)
    for e in [1.0, 1.1, 1.2, 1.3]:
        x_points = detect_points(img_mask, preds_train_t, eps=e, is_x_axis=True, need_display=True)
        y_points = detect_points(img_mask, preds_train_t, eps=e, is_x_axis=False, need_display=True)
        x_scale = calc_axis_scale(x_points, x_vals, [texts_pos[i] for i in x_vals_idx], is_x_axis=True)
        y_scale = calc_axis_scale(y_points, y_vals, [texts_pos[i] for i in y_vals_idx], is_x_axis=False)
        print('x_scale, y_scale', x_scale, y_scale)
        if abs(x_scale[0]) > 0 and abs(y_scale[0]) > 0:
            print('non zero scale found')
            break

    for p in x_points:
        cv2.rectangle(img_mask, (p[0] - 10, p[1] - 10), (p[0] + 10, p[1] + 10), (255, 0, 0), -1)
    for p in y_points:
        cv2.rectangle(img_mask, (p[1] - 10, p[0] - 10), (p[1] + 10, p[0] + 10), (255, 0, 0), -1)
    print('x  scale:', x_scale)
    print('y  scale:', y_scale)

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(0, 7):
        ax[i // 3, i % 3].imshow(np.asarray(preds_train_t[0, i, :, :], dtype=np.int))
    plt.show()
    return title_vals, xlabel, ylabel, x_scale, y_scale

def flattern_list(lst):
    '''
    For nested lists return a list with flattern values
    @param lst: source list
    @return: flattered list
    '''
    res = []
    for l in lst:
        if isinstance(l, list):
            res += flattern_list(l)
        else:
            res.append(l)
    return res

def recognize(input_img, config):
    '''
    Full plot recognition
    @param input_img: source image
    '''
    orig_shape = max(input_img.shape[0], input_img.shape[1])
    scale_orig = 512 / orig_shape

    #padding = 10
    #input_img = input_img[padding:input_img.shape[0] - padding, padding:input_img.shape[1] - padding]
    title_vals, xlabel, ylabel, x_scale, y_scale = recognize_axis(input_img, config)
    plots = pt.recognize_plot(input_img, True)
    realplots = []
    display_data = []
    df = pd.DataFrame()
    colors = ['r-', 'g-', 'b-', 'k-', 'm-', 'c-']
    for i, plot in enumerate(plots):
        x_vals, y_vals, y2_vals = [], [], []
        if abs(x_scale[0]) > 0:
            for x in plot[0]:
                x_vals.append((x * scale_orig - x_scale[1]) / x_scale[0])
        else:
            x_vals = plot[0]
        if abs(y_scale[0]) > 0:
            for y2 in plot[2]:
                y2_vals.append((y2 * scale_orig - y_scale[1]) / y_scale[0])
        else:
            y2_vals = plot[2]
        y2_vals = np.array(y2_vals)
        y2_vals = np.max(y2_vals) - y2_vals #+ np.min(y2_vals)
        fig, ax = plt.subplots()
        ax.plot(x_vals, y2_vals, colors[i])
        if len(title_vals) > 0:
            ax.set_title(" ".join(flattern_list(title_vals)))

        if len(xlabel) > 0:
            ax.set_xlabel(" ".join(flattern_list(xlabel)))
        if len(ylabel) > 0:
            ax.set_ylabel(" ".join(flattern_list(ylabel)))
        plt.show()
    print(df.head())
    pt.show_texts(input_img, config)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("settings.ini")

    input_img = cv2.imread(config['storage']['filename'])
    recognize(input_img, config)
