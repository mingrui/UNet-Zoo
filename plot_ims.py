import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
# outs = np.load('Numpy-batch-7-outs.npy')
# masks = np.load('Numpy-batch-7-masks.npy')

curr_pos = 0

def plot_test():
    final_outs = []
    final_masks = []
    final_images = []

    loss = np.load('npy-files/loss-files/OutMasks-UNetSmall_Loss_bs=5_ep=10_lr=0.001.npy')
    base_name = 'OutMasks-unetsmall'

    out_folder = '/mnt/960EVO/workspace/UNet-Zoo/npy-files/out-files/'
    f_lst = os.listdir(out_folder)
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

def plot_pred():
    save_dir = '/mnt/DATA/datasets/Pathology/Necrosis_Segmentation/pred'
    final_outs = []
    final_images = []

    loss = np.load('npy-files/loss-files/OutMasks-UNetSmall_Loss_bs=5_ep=10_lr=0.001.npy')
    base_name = 'OutMasks-unetsmall'

    out_folder = '/mnt/960EVO/workspace/UNet-Zoo/npy-files/out-files/'
    f_lst = os.listdir(out_folder)
    f_count = int(len(f_lst)/2)

    print('file count : ', f_count)

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
            ax.imshow(ax_params[curr_pos][1], cmap=cm.autumn, alpha=0.2)
            fig.canvas.draw()
            return
        else:
            return
        curr_pos = curr_pos % len(ax_params)

        ax.cla()
        ax.imshow(ax_params[curr_pos][0], cmap=cm.gray)
        ax.imshow(ax_params[curr_pos][1], cmap=cm.autumn, alpha=0.2)
        fig.canvas.draw()

    for i in range(f_count):
        batches = final_images[i].shape[0]
        print(batches)

        ax_params = []

        for s in range(batches):
            plt1 = np.squeeze(final_images[i][s, :, :])
            plt3 = np.squeeze(final_outs[i][s, :, :])

            image_data = plt1
            segment_data = np.ma.masked_where(plt3 < 0.9, plt3)

            ax_params.append([image_data, segment_data])

        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', key_event)
        ax = fig.add_subplot(111)
        ax.imshow(ax_params[0][0], cmap=cm.gray)
        ax.imshow(ax_params[0][1], cmap=cm.autumn, alpha=0.2)
        # max window
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        # plt.show()

        for idx, ax_param in enumerate(ax_params):
            ax.cla()
            ax.imshow(ax_param[0], cmap=cm.gray)
            ax.imshow(ax_param[1], cmap=cm.autumn, alpha=0.2)
            fig.canvas.draw()
            plt.savefig(os.path.join(save_dir, str(i)+'_'+str(idx)+'.png'))

# print(final_outs.shape)
# print(final_masks.shape)
#
# sio.savemat('mat-files/final_outputs.mat', {'data': final_outs})
# sio.savemat('mat-files/final_masks.mat', {'data': final_masks})
# sio.savemat('mat-files/final_images.mat', {'data': final_images})
#
# for i in range(len(outs)):
#     plt1 = 255 * np.squeeze(outs[i][:, :, :]).astype('uint8')
#     plt2 = 255 * np.squeeze(masks[i][:, :, :]).astype('uint8')
#     plt3 = 255 * np.squeeze(images[i][:, :, :]).astype('uint8')
#
#     print(plt1, plt2, plt3)
#
#     plt.subplot(1, 3, 1)
#     plt.imshow(plt1, cmap='gray')
#     plt.title("UNet Out")
#
#     plt.subplot(1, 3, 2)
#     plt.imshow(plt2, cmap='gray')
#     plt.title("Mask")
#
#     plt.subplot(1, 3, 3)
#     plt.imshow(plt3, cmap='Blues')
#     plt.title('Image')
#
#     plt.show()

if __name__ == '__main__':
    # plot_test()
    plot_pred()
