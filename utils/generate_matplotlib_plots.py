import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import json

def gen_xlabel():
    return ['xlabel']

def gen_ylabel():
    return ['ylabel']

def gen_titlelabel():
    return ['title label']

def generate_param_space():
    param_vals = {
        'color': ['r-', 'g-'], #, 'b-', 'y-', 'c-', 'm-', 'k-'],
        'title_fontdict': [
            {
                'fontsize': 10,
                'fontweight': 'bold',
                'color': 'red'
            },
        #     {
        #         'fontsize': 12,
        #         'fontweight': 'bold',
        #         'color': 'green'
        #     },
        #     {
        #         'fontsize': 14,
        #         'fontweight': 'normal',
        #         'color': 'blue'
        #     },
        #     {
        #         'fontsize': 16,
        #         'fontweight': 'normal',
        #         'color': 'gray'
        #     },
        #     {
        #         'fontsize': 14,
        #         'fontweight': 'bold',
        #         'color': 'yellow'
        #     }
        ],
        'title_loc': ['center', 'left', 'right'],
        'title_pad': [2.0, 4.0],
        'need_grid': [True, False],
        'x': [
            # np.linspace(0, 100, vals_number).tolist(),
            # np.linspace(-100, 100, vals_number).tolist(),
            np.linspace(-2, 3, 30).tolist(),
            np.linspace(0, 1, 20).tolist(),
            # np.linspace(-3, 10, vals_number).tolist()
        ],
        'y': [
            np.sin,
            # np.tan,
            lambda x: 2 * x + 5,
            lambda x: 0.01 * x ** 2 - 50
        ],
        'xlabel': gen_xlabel(),
        'ylabel': gen_ylabel(),
        'title_label': gen_titlelabel()
    }
    return param_vals

def combinate_params(param_vals):
    '''
    Generates full combinations of params from list of dictionaries
    :param param_vals:
    :return:
    '''
    target_params = []
    for key, vals in param_vals.items():
        if vals is None:
            continue
        # multiply number of rows of existing list to number of new params
        new_target = []
        for val in vals:
            if len(target_params) > 0:
                for p in target_params:
                    new_row = p.copy()
                    new_row.update({key: val})
                    new_target.append(new_row)
            else:
                new_target.append({key: val})
        target_params = new_target

    return target_params

def run_generation(params):
    print(f'number of generated images: {len(params)}')
    for i, param in enumerate(params):
        plt.figure(i)
        plt.ioff()
        plt.plot(param['x'], param['y'](np.array(param['x'])), param['color'])
        plt.xlabel(param['xlabel'])
        plt.ylabel(param['ylabel'])
        plt.grid(param['need_grid'])
        plt.title(label = param['title_label'], fontdict = param['title_fontdict'], loc = param['title_loc'], pad = param['title_pad'])
        param['y'] = param['y'].__name__
        plt.savefig('../data/generated_matplotlib/plots/plot_{:0>4d}.png'.format(i))

    with open('../data/generated_matplotlib/matplotlib_data_labels.json', 'w') as fout:
        json.dump(params, fout)

if __name__ == "__main__":
    params_space = generate_param_space()
    gen_params = combinate_params(params_space)
    run_generation(gen_params)
