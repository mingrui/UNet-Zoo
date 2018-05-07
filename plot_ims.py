import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
# outs = np.load('Numpy-batch-7-outs.npy')
# masks = np.load('Numpy-batch-7-masks.npy')
from PIL import Image

curr_pos = 0

def plot_test(base_name):
    final_outs = []
    final_masks = []
    final_images = []

    out_folder = '/mnt/960EVO/workspace/UNet-Zoo/npy-files/out-files/'
    f_lst = glob.glob(out_folder+base_name+'-batch*')
    f_lst.sort()
    # print(f_lst)
    f_count = int(len(f_lst)/3)

    print('file count : ', f_count)

    for i in range(f_count):
        outs = np.load(
            'npy-files/out-files/'+base_name+'-batch-{}-outs.npy'.format(i))
        masks = np.load(
            'npy-files/out-files/'+base_name+'-batch-{}-masks.npy'.format(i))
        images = np.load(
            'npy-files/out-files/'+base_name+'-batch-{}-images.npy'.format(i))

        final_outs.append(outs)
        final_masks.append(masks)
        final_images.append(images)

    final_outs = np.asarray(final_outs)
    final_masks = np.asarray(final_masks)
    final_images = np.asarray(final_images)

    print('final images shape : ', final_images.shape)

    def key_event(e):
        global curr_pos

        if e.key == 'right':
            curr_pos += 1
        elif e.key == 'left':
            curr_pos -= 1
        elif e.key == 'up':
            # turn off overlay
            ax.cla()
            ax.imshow(ax_params[curr_pos][0], cmap=cm.gray)
            fig.canvas.draw()
            return
        elif e.key == 'down':
            # turn on overlay
            ax.cla()
            ax.imshow(ax_params[curr_pos][0], cmap=cm.gray)
            ax.imshow(ax_params[curr_pos][1], cmap=cm.jet, alpha=0.3)
            ax.imshow(ax_params[curr_pos][2], cmap=cm.autumn, alpha=0.2)
            fig.canvas.draw()
            return
        else:
            return
        curr_pos = curr_pos % len(ax_params)

        ax.cla()
        ax.imshow(ax_params[curr_pos][0], cmap=cm.gray)
        ax.imshow(ax_params[curr_pos][1], cmap=cm.jet, alpha=0.3)
        ax.imshow(ax_params[curr_pos][2], cmap=cm.autumn, alpha=0.2)
        fig.canvas.draw()

    for i in range(f_count):
        slices = final_images[i].shape[0]

        ax_params = []

        for s in range(slices):
            plt1 = np.squeeze(final_images[i][s, :, :])
            plt2 = np.squeeze(final_masks[i][s, :, :])
            plt3 = np.squeeze(final_outs[i][s, :, :])

            image_data = plt1
            masked_data = np.ma.masked_where(plt2 < 0.9, plt2)
            segment_data = np.ma.masked_where(plt3 < 0.9, plt3)

            ax_params.append([image_data, masked_data, segment_data])

        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', key_event)
        ax = fig.add_subplot(111)
        ax.imshow(ax_params[0][0], cmap=cm.gray)
        ax.imshow(ax_params[0][1], cmap=cm.jet, alpha=0.3)
        ax.imshow(ax_params[0][2], cmap=cm.autumn, alpha=0.2)
        # max window
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

def plot_pred_backup(file_names, segmentation_prediction_dir, base_name, outmask_dir, has_test_set = False):
    final_outs = []
    final_images = []
    final_masks = []

    print(os.path.join(outmask_dir, base_name + '-batch*'))
    f_lst = glob.glob(os.path.join(outmask_dir, base_name + '-batch*'))
    f_lst.sort()
    if has_test_set:
        num = 3
    else:
        num = 2
    f_count = int(len(f_lst)/num)

    print('file count : ', f_count)
    # f_count = 8
    for i in range(f_count):
        outs = np.load(
            os.path.join(outmask_dir, base_name+'-batch-{}-outs.npy'.format(i)))
        images = np.load(
            os.path.join(outmask_dir, base_name+'-batch-{}-images.npy'.format(i)))
        if has_test_set:
            masks = np.load(
                os.path.join(outmask_dir,base_name + '-batch-{}-masks.npy'.format(i)))

        final_outs.append(outs)
        final_images.append(images)
        if has_test_set:
            final_masks.append(masks)

    final_outs = np.asarray(final_outs)
    final_images = np.asarray(final_images)
    if has_test_set:
        final_masks = np.asarray(final_masks)

    print('final images shape : ', final_images.shape)

    count = 0
    # contains files with the same prefix, the same patient file
    merge_dict = {}
    for i in range(f_count):
        batches = final_images[i].shape[0]
        print(batches)

        ax_params = []

        for s in range(batches):
            plt1 = np.squeeze(final_images[i][s, :, :])
            plt3 = np.squeeze(final_outs[i][s, :, :])
            if has_test_set:
                plt2 = np.squeeze(final_masks[i][s, :, :])

            image_data = plt1
            segment_data = np.ma.masked_where(plt3 < 0.9, plt3)

            if has_test_set:
                masked_data = np.ma.masked_where(plt2 < 0.9, plt2)

            if has_test_set:
                ax_params.append((image_data, segment_data, file_names[count], masked_data))
            else:
                ax_params.append((image_data, segment_data, file_names[count]))
                np.savetxt(os.path.join(segmentation_prediction_dir, file_names[count]+'.txt'), segment_data[1].astype(int), fmt='%i',delimiter=',')

            count += 1

        for idx, ax_param in enumerate(ax_params):
            print(ax_param[2])

            fig = plt.figure(num=None, figsize=(4, 4), dpi=128, facecolor='w', edgecolor='k')
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(ax_param[0], cmap=cm.gray)
            if has_test_set:
                ax.imshow(ax_param[3], cmap=cm.jet, alpha=0.3)
            ax.imshow(ax_param[1][1], cmap=cm.autumn, alpha=0.2)
            plt.savefig(os.path.join(segmentation_prediction_dir, ax_param[2]))
    print(count)

def plot_pred(file_names, segmentation_prediction_dir, base_name, outmask_dir, has_test_set = False):
    final_outs = []
    final_images = []
    final_masks = []

    print(os.path.join(outmask_dir, base_name + '-batch*'))
    f_lst = glob.glob(os.path.join(outmask_dir, base_name + '-batch*'))
    f_lst.sort()
    if has_test_set:
        num = 3
    else:
        num = 2
    f_count = int(len(f_lst)/num)

    print('file count : ', f_count)
    # f_count = 8
    for i in range(f_count):
        outs = np.load(
            os.path.join(outmask_dir, base_name+'-batch-{}-outs.npy'.format(i)))
        images = np.load(
            os.path.join(outmask_dir, base_name+'-batch-{}-images.npy'.format(i)))
        if has_test_set:
            masks = np.load(
                os.path.join(outmask_dir,base_name + '-batch-{}-masks.npy'.format(i)))

        final_outs.append(outs)
        final_images.append(images)
        if has_test_set:
            final_masks.append(masks)

    final_outs = np.asarray(final_outs)
    final_images = np.asarray(final_images)
    if has_test_set:
        final_masks = np.asarray(final_masks)

    print('final images shape : ', final_images.shape)

    count = 0
    # contains files with the same prefix, the same patient file
    merge_dict = {}
    for i in range(f_count):
        batches = final_images[i].shape[0]
        print(batches)

        ax_params = []

        for s in range(batches):
            plt1 = np.squeeze(final_images[i][s, :, :])
            plt3 = np.squeeze(final_outs[i][s, :, :])
            if has_test_set:
                plt2 = np.squeeze(final_masks[i][s, :, :])

            image_data = plt1
            segment_data = np.ma.masked_where(plt3 < 0.9, plt3)

            if has_test_set:
                masked_data = np.ma.masked_where(plt2 < 0.9, plt2)

            if has_test_set:
                ax_params.append([image_data, segment_data, file_names[count], masked_data])
            else:
                ax_params.append([image_data, segment_data, file_names[count]])
                np.savetxt(os.path.join(segmentation_prediction_dir, file_names[count]+'.txt'), segment_data[1].astype(int), fmt='%i',delimiter=',')

            count += 1

        for idx, ax_param in enumerate(ax_params):
            print(ax_param[2])

            fig = plt.figure(num=None, figsize=(4, 4), dpi=128, facecolor='w', edgecolor='k')
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax_param[0] = np.dstack(ax_param[0])
            ax.imshow(ax_param[0], cmap=cm.gray)
            if has_test_set:
                ax.imshow(ax_param[3], cmap=cm.jet, alpha=0.3)
            ax.imshow(ax_param[1][1], cmap=cm.autumn, alpha=0.5)
            plt.savefig(os.path.join(segmentation_prediction_dir, ax_param[2]))
    print(count)
