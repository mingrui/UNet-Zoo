import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import math
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
# outs = np.load('Numpy-batch-7-outs.npy')
# masks = np.load('Numpy-batch-7-masks.npy')

curr_pos = 0

def plot_test():
    final_outs = []
    final_masks = []
    final_images = []

    loss = np.load('npy-files/loss-files/OutMasks-bdclstm_bs=1_ep=5_lr=0.001.npy')
    base_name = 'OutMasks'

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

def plot_pred(file_names, save_dir, loss_file, base_name, out_folder):
    final_outs = []
    final_images = []

    loss = np.load(loss_file)

    f_lst = glob.glob(out_folder + base_name + '-batch*')
    f_lst.sort()
    f_count = int(len(f_lst)/2)

    print('file count : ', f_count)
    # f_count = 8
    for i in range(f_count):
        outs = np.load(
            'npy-files/out-files/'+base_name+'-batch-{}-outs.npy'.format(i))
        images = np.load(
            'npy-files/out-files/'+base_name+'-batch-{}-images.npy'.format(i))

        final_outs.append(outs)
        final_images.append(images)

    final_outs = np.asarray(final_outs)
    final_images = np.asarray(final_images)

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

            image_data = plt1
            segment_data = np.ma.masked_where(plt3 < 0.9, plt3)

            ax_params.append((image_data, segment_data, file_names[count]))
            x = file_names[count].split('.')[0].split('_')
            key = '_'.join(x[:-2])
            if key in merge_dict:
                merge_dict[key].append(file_names[count])
            else:
                merge_dict[key] = [file_names[count]]
            count += 1

        for idx, ax_param in enumerate(ax_params):
            print(ax_param[2])

            fig = plt.figure(num=None, figsize=(4, 4), dpi=128, facecolor='w', edgecolor='k')
            ax = plt.Axes(fig, [0, 0, 1, 1])
            ax.set_axis_off()
            fig.add_axes(ax)
            # print(ax_param[0].shape)
            ax.imshow(ax_param[0], cmap=cm.gray)
            # print(ax_param[1].shape)
            ax.imshow(ax_param[1], cmap=cm.autumn, alpha=0.2)
            plt.savefig(os.path.join(save_dir, ax_param[2]))
    print(count)

    def numeric_sort_lambda(x):
        x = x.split('.')[0]
        if 'valid' in x:
            return int(x.split('_')[4])
        else:
            return int(x.split('_')[3])

    # merge photos
    os.chdir(save_dir)
    for key in merge_dict:
        merge_dict[key].sort(key=numeric_sort_lambda)

        sqrt = len(merge_dict[key]) ** (1/2.0)
        x = math.ceil(sqrt)

        merge_images(merge_dict[key], (x,x)).save(key+'.png')
        print(key+'.png')

        for image in  merge_dict[key]:
            os.remove(image)

def merge_images(img_list, dim):
    img = img_list[0]
    row_n = dim[0]
    col_n = dim[1]

    img = Image.open(img)
    (w, h) = img.size

    canvas_size = (w * row_n, h * col_n)
    result = Image.new('RGB', canvas_size)

    for idx, img in enumerate(img_list):
        box = (w*int(idx%col_n), h*int(idx/row_n))
        # print(img)
        # print(box)
        img = Image.open(img)
        result.paste(im=img, box=box)

    return result

if __name__ == '__main__':
    plot_test()
    #plot_pred()
