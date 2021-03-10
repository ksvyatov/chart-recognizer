import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img
from tensorflow.python.keras.models import load_model
from matplotlib import pyplot as plt
import cv2
import math
import recognizer.plots as pt

print(cv2.__version__)

n_channels = 3
CLASS_NAMES = ["Title", "xdata", "ydata", "xlabel", "ylabel", "xpoints", "ypoints"]


def resize_image(image, size = 512, inter = cv2.INTER_AREA):
    orig_shape = max(image.shape[0], image.shape[1])
    new_img = np.ones((orig_shape, orig_shape, image.shape[2]), np.uint8) * 255
    new_img[:image.shape[0], :image.shape[1], :3] = image
    return cv2.resize(new_img, (size, size), interpolation=inter)

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def filter_points(points):
    """
    Detect points that are located on the same distance between each other
    """
    distances = []
    # get all distances between points
    if len(points) > 1:
        for i, point in enumerate(points):
            d = abs(points[i][0] - points[i + 1][0])
            if d > 512 / 40:
                distances.append((points[i][0], d))
            # last point also in any case
            if i >= len(points) - 2: 
                distances.append((points[i + 1][0], d))
                break
    #print(distances)
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
    #print('fr_dist', fr_dist)
    max_pos, imax_pos = 0, 0
    for i, a in enumerate(fr_dist):
        if a[1] >= max_pos:
            max_pos = a[1]
            imax_pos = i
    #print('dist', fr_dist[imax_pos])
    
    new_dist = []
    for i, point in enumerate(distances):
        print(point[1], fr_dist[imax_pos][0])
        if abs(point[1] - fr_dist[imax_pos][0]) < 5:
            new_dist.append(point)
    new_dist.append((new_dist[-1][0] + fr_dist[imax_pos][0], fr_dist[imax_pos][0]))
    return new_dist#, fr_dist[imax_pos]

def detect_points(src_img, preds, is_x_axis=True, need_display=False):
    """
    Detect points for plot scaling 
    """
    # алгоритм: 1. получить координаты рисок, 2. получить численные значения по оси x, y 3. определить для каждой риски значения из 2 пункта 4.  

    img = src_img.copy()
    # channel for x points or y points accordingly
    print('preds shape', preds.shape)
    xpoints = preds[0, 5, :, :] if is_x_axis else preds[0, 6, :, :]
    print('xpoints shape', xpoints.shape)
    
    # find largest contour among all areas
    #contours, hier = cv2.findContours(xpoints, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hier = cv2.findContours(xpoints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    max_area = 0
    x, y, w, h = 0, 0, 0, 0
    for c in contours:
        xc, yc, wc, hc = cv2.boundingRect(c)
        if (wc * hc) > max_area:
            x, y, w, h = xc, yc, wc, hc
            max_area = wc * hc
    
    if (w * h) > 100:
        print('area: ', x, y, w, h)
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                if not (x < i < x + w and y < j < y + h):
                    img[j, i] = [0, 0, 0]
                else:
                    img[j, i] = img[j, i] * 255
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
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
            #cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
    
    # among all lines, find a line in the middle of the area of interest
    print(x_vals)
    res = []
    if len(x_vals) > 0:
        x_idx = (np.abs(x_vals - np.mean(x_vals))).argmin()
        y_idx = (np.abs(y_vals - np.mean(y_vals))).argmin()
        x_line = int(x_vals[x_idx])
        y_line = int(y_vals[y_idx])

        # sliding window for finding crosses
        if is_x_axis:
            x_points = [(np.mean(img[y_line-7 : y_line+7, i] / 255)) for i in range(x, x + w)]
            res = [(i, pos) for i, pos in enumerate(x_points) if pos < np.mean(x_points)]
            if need_display:
                for i in filter_points(res):
                    cv2.line(img, (int(x + i[0]), 0), (int(x + i[0]), 512), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(img, (0, y_line), (512, y_line), (255, 0, 0), 1, cv2.LINE_AA)
        else:
            y_points = [(np.mean(img[i, x_line-9 : x_line+9] / 255)) for i in range(y, y + h)]
            res = [(i, pos) for i, pos in enumerate(y_points) if pos < np.mean(y_points)]
            if need_display:
                for i in filter_points(res):
                    cv2.line(img, (0, int(y + i[0])), (512, int(y + i[0])), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.line(img, (x_line, 0), (x_line, 512), (255, 0, 0), 1, cv2.LINE_AA)
    
    if need_display:
        plt.figure(figsize = (10,10))
        plt.imshow(img)
        plt.plot()
    return res

def recognize_axis(input_img):
    model = load_model('model-plots_bw-30.h5', custom_objects={'mean_iou': mean_iou}, compile=False)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #optimizer = tf.keras.optimizers.Adam(lr=1e-6)
    optimizer = tf.keras.optimizers.SGD(lr=1e-7)
    metrics = [tf.keras.metrics.categorical_accuracy]
    model.compile(optimizer, loss, metrics)
    #print(model.summary())

    input_img = resize_image(input_img)
    if n_channels == 1:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = cv2.adaptiveThreshold(input_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 12)
    print(input_img.shape)
    plt.figure(figsize = (10,10))
    plt.imshow(np.asarray(input_img, dtype=np.int))
    plt.show()
    input_img = np.asarray(input_img, dtype=np.float32) // 255
    input_img = np.asarray(input_img, dtype=np.uint8) 
    #X = np.zeros((1, 512, 512, n_channels))
    X = np.zeros((1, n_channels, 512, 512))
    for i in range(n_channels):
        X[0, i, :, :] = input_img[:,:,i]

    preds_train = model.predict(X, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_test_upsampled = []
    #print(f'preds len {len(preds_train)}')
    for i in range(len(preds_train)):
         preds_test_upsampled.append(np.squeeze(preds_train[i]))
            
    print(preds_train_t[0, 0, :, :].shape)
    detect_points(input_img, preds_train_t, is_x_axis=True, need_display=True)
    detect_points(input_img, preds_train_t, is_x_axis=False, need_display=True)
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(0, 7):
        ax[i // 3, i % 3].imshow(np.asarray(preds_train_t[0, i, :, :], dtype=np.int))
    plt.show()
    
if __name__ == "__main__":
    input_img = cv2.imread('data/recognizer/imgs/resized/test04.png')
    recognize_axis(input_img)

