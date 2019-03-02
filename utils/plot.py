import os
from glob import glob

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def comute_mAP_splitted(path='/data/104-1/Experiments/Faster_RCNN/models', dataset='kitti', net='res101',
                        classes=['car', 'pedestrian'], scores=['easy', 'moderate', 'hard'], n_thresholds=41,
                        subeval=None):
    n_classes = len(classes)
    n_scores = len(scores)

    if subeval is not None:
        print('evaluation file: %s' % subeval)
        path_prefix = os.path.join(path, 'test', dataset, subeval, net)
    else:
        path_prefix = os.path.join(path, 'test', dataset, net)
    epochs = glob(os.path.join(path_prefix, 'model_iter*/plot'))

    epochs = np.sort(np.asarray([int(epoch.split('/')[-2].split('_')[1].split('r')[1]) for epoch in epochs]))

    if len(epochs) == 0:
        return []

    n_epochs = max(epochs)

    matrix_scores = -1 * np.ones((n_epochs, n_classes, n_scores, n_thresholds))

    for epoch in epochs:
        for indx_class, eval_class in enumerate(classes):
            if os.path.exists(
                    os.path.join(path_prefix, 'model_iter' + str(epoch), 'plot', eval_class + '_detection.txt')):
                filename = os.path.join(path_prefix, 'model_iter' + str(epoch), 'plot',
                                        eval_class + '_detection.txt')
                with open(filename, 'r') as f:
                    v_lines = f.readlines()
                    v_lines = np.asarray([[float(number) for number in line.rstrip().split(' ')] for line in v_lines])
                    v_lines = np.transpose(v_lines)
                matrix_scores[epoch - 1, indx_class, 0, :] = np.transpose(v_lines[2, :])
                matrix_scores[epoch - 1, indx_class, 1, :] = np.transpose(v_lines[1, :])
                matrix_scores[epoch - 1, indx_class, 2, :] = np.transpose(v_lines[3, :])

    mean_scores = np.mean(matrix_scores, axis=3)
    return mean_scores


def plot_mIoU(vmIoU, legends, classes, scores, name_prefix):
    scores = []
    for indx_score, value_score in enumerate(scores):
        accum_data_y = None
        accum_data_x = None
        print(value_score)  # easy, moderate, hard
        for indx_class, value_class in enumerate(classes):
            data_y = [np.asarray(mIoU)[:, indx_class, indx_score] if len(mIoU) > 0 else [] for mIoU in vmIoU]
            best_data = [max(data) if len(mIoU) > 0 else 0 for data in data_y]
            lens_data = [len(data) for data in data_y]
            data_x = [range(1, len(data) + 1) for data in data_y]
            filename = name_prefix + '_plot_' + value_class + '_' + value_score + '.png'
            title = name_prefix.split('/')[-1] + '_' + value_class + '_' + value_score
            # plot_multiple_data(data_y, data_x, filename, title, legends)
            if accum_data_y is None and accum_data_x is None:
                accum_data_y = data_y
                accum_data_x = data_x
            else:
                accum_data_y = [[el1 + el2 for (el1, el2) in zip(vX, ac_vX)] for (vX, ac_vX) in zip(data_y, accum_data_y)]
                accum_data_x = [[el1 + el2 for (el1, el2) in zip(vX, ac_vX)] for (vX, ac_vX) in zip(data_x, accum_data_x)]

        if accum_data_x is not None and accum_data_y is not None:
            filename = name_prefix + '_plot_mean_' + value_score + '.png'
            title = name_prefix.split('/')[-1] + '_mean_' + value_score
            accum_data_x = [[el / len(classes) for el in v_x] for v_x in accum_data_x]
            accum_data_y = [[el / len(classes) for el in v_x] for v_x in accum_data_y]
            # plot_multiple_data(accum_data_y, accum_data_x, filename, title, legends)

            if indx_score == 0:
                best_indx = [np.argmax(np.asarray(data)) for data in accum_data_y]
                best_epoch = [x[indx] for (x, indx) in zip(accum_data_x, best_indx)]
            best_value = [y[indx] for (y, indx) in zip(accum_data_y, best_indx)]
            # print(legends) # Model name
            print(best_epoch)  # best epoch model
            # print(best_value) # best mean mAP  over each class

            for indx_class, value_class in enumerate(classes):
                data_y = [np.asarray(mIoU)[:, indx_class, indx_score] if len(mIoU) > 0 else [] for mIoU in vmIoU]
                value = [y[indx] for (y, indx) in zip(data_y, best_indx)]
                print(value_class)  # car, pedestrian, cyclist
                print(value)  # best mAP per class
                scores.append([value_class, value])
    return scores


def plot_multiple_data(data_y, data_x, filename, title, legends):
    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_y = 100
    max_y = 0
    for idx, _ in enumerate(data_y):
        x = np.asarray([data for data in data_x[idx] if data_y[idx][data - 1] != -1])
        y = np.asarray([data for data in data_y[idx] if data != -1])
        max_y = max(max(y + 0.05), max_y)
        min_y = min(min(y), min_y)
        lines = ax.plot(x, 100 * y)
    ax.set_title(title)

    lgd = ax.legend(legends, loc='center right', bbox_to_anchor=(2.5, 0.5))

    ax.set_xlabel('Iteration')
    ax.set_ylabel('IoU')
    ax.set_ylim([min_y * 100.0, max_y * 100.0])
    # ax.set_ylim([1,100])

    plt.grid()

    # fig.draw()
    fig.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def compute_plot(cf, subeval=None):
    path = os.path.join(cf.exp_folder)  # '/data/104-1/Experiments/Detectron/Models'
    path_out = os.path.join(cf.exp_folder, '..', 'plots')  # '/data/104-1/Experiments/Detectron/plots'
    # Nets
    v_source_db = [cf.dataset]
    v_nets = [cf.model_type]

    v_models = [cf.model_name]
    if subeval is not None:
        print(subeval)

    title_prefix = 'baselines_nets_synthia_random_generator'

    classes = cf.labels
    scores = ['moderate', 'easy', 'hard']

    n_thresholds = 41

    path_out = os.path.join(path_out, title_prefix)
    # if not os.path.exists(path_out):
    #     os.makedirs(path_out)

    vmIoU = []
    legends = []
    for model in v_models:
        model_path = os.path.join(path, model + '_' + cf.dataset)
        for net in v_nets:
            for sourceDB in v_source_db:
                mean_scores = comute_mAP_splitted(path=model_path, dataset=sourceDB, net=net, classes=classes,
                                                  scores=scores, n_thresholds=n_thresholds, subeval=subeval)
                if mean_scores:
                    vmIoU.append(mean_scores)
                    legends.append(os.path.join(model, net, sourceDB))
    scores = plot_mIoU(vmIoU, legends, classes, scores, os.path.join(path_out, title_prefix))
    return scores
