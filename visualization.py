
import numpy as np
import matplotlib.pyplot as plt


def imshow(img):
    x = img - np.min(img)
    x = x / np.max(x)
    plt.imshow(np.transpose(x, (1, 2, 0)))
    plt.axis('off')


def plot_meis(activations, scores, inputs, filename, sort=True,
              num_show=10, show_all=False, plot=False):
    # show top5 and bottom5 scores
    # show by decreasing activations
    if show_all:    
        num_unit = activations.shape[1]
    else:
        num_unit = num_show
    if len(scores) > 0:
        if sort:
            ind_sort = np.argsort(scores)
            ind_units = np.concatenate([
                ind_sort[:num_unit // 2], ind_sort[-num_unit // 2 - 1:]
            ])
        else:
            ind_units = np.arange(num_unit)
    elif scores == None:
        scores = activations.mean(0)
        ind_units = np.arange(num_unit)
    plt.figure(figsize=((num_show + 1) * 1.1, num_unit * 1.1))
    for i in range(num_unit):
        ind_unit = ind_units[i]
        y = activations[:, ind_unit].copy()
        plt.subplot(num_unit, num_show + 1, 1 + i * (num_show + 1))
        plt.axis('off')
        plt.text(0, 0, 'Unit: %s,\nScore: %.4f,\nMean: %.4f,\nStd: %.4f' % (
            ind_unit, scores[ind_unit], y.mean(), y.std()
        ))
        ind_sort = np.argsort(y)[::-1]
        for j in range(num_show):
            plt.subplot(num_unit, num_show + 1, j + 2 + i * (num_show + 1))
            imshow(inputs[ind_sort[j]])
            plt.title('%.4f' % y[ind_sort[j]])
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig(filename)
        plt.clf()