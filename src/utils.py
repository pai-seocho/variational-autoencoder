import math
import matplotlib.pyplot as plt
import random


def plot_images(imgs, names=None, random_order=False, savepath=None):
    h = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure()
    plt.gcf().set_size_inches((20, 20))
    for i in range(len(imgs)):
        ax = fig.add_subplot(h, h, i + 1)
        if random_order:
            ind = random.randint(0, len(imgs) - 1)
        else:
            ind = i
        img = imgs[ind]
        plt.axis('off')
        plt.imshow(img)
        #
        if not names is None:
            ax.set_title(str(names[ind]))
    if not savepath is None:
        plt.savefig(savepath)
    plt.tight_layout()
    plt.show()
