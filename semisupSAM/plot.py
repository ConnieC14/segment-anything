
import matplotlib.pyplot as plt
import numpy as np


def show_mask(mask, ax, random_color=False, color=[]):
    """

    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif len(color) > 0:
        pass
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    return ax


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )
    
    return ax

def plot_comparison(image, boxes, mask2D, medsam_pred):
    """
        idx: batch layer if there are batches
    """
    rows = image.shape[0]
    fig, axs = plt.subplots(rows, 3, figsize=(25, 25))

    for idx in range(rows):
        axs[idx, 0].imshow(image[idx,[0],:,:].cpu().permute(1, 2, 0).numpy())
        if (boxes.sum() > 0):
            show_box(boxes[idx,:].numpy(), axs[idx, 0])
        axs[idx, 0].axis("off")
        # set title
        axs[idx, 0].set_title('Image')

        axs[idx, 1].imshow(image[idx,[0],:,:].cpu().permute(1, 2, 0).numpy())
        show_mask(mask2D[idx,:,:,:].cpu().numpy(), 
                  axs[idx, 1],
                  random_color=False,
                  color = np.array([1, 0.5, 0.5, 0.8]))
        if (boxes.sum() > 0):
            show_box(boxes[idx,:].numpy(), axs[idx, 1])
        axs[idx, 1].axis("off")
        # set title
        axs[idx, 1].set_title('Truth Mask')

        axs[idx, 2].imshow(image[idx,[0],:,:].cpu().permute(1, 2, 0).numpy())
        show_mask(medsam_pred[idx,:,:,:].detach().cpu().numpy(), 
                  axs[idx, 2],
                  random_color=False,
                  color = np.array([1, 0.5, 0.5, 0.8]))
        if (boxes.sum() > 0):
            show_box(boxes[idx,:].numpy(), axs[idx, 2])
        axs[idx, 2].axis("off")
        # set title
        axs[idx, 2].set_title('Pred Image')

    plt.subplots_adjust(wspace=0.01, hspace=0)
    #plt.savefig('test_training_output_1.png')

    return fig
    
def plot_loss(losses, save_name):
    plt.plot(losses)
    plt.title("Dice + Cross Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_name)
    plt.close()