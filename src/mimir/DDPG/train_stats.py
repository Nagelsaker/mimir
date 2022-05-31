import numpy as np
import os, sys

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection

def visualizeTraining(name):
    import matplotlib.pyplot as plt
    path_to_metadata = f"{os.path.dirname(os.path.abspath(__file__))}/meta_data/success_reward{name}.npy"
    meta = np.load(path_to_metadata)
    
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(15, 5)
    fig.set_dpi(80)

        # Compensating for error when saving rewards
    
    axs[0].plot(meta[:,0])
    axs[0].legend(["s"])
    axs[0].set(ylabel="Success rate")
    axs[0].set_title("Validation Success Rate")
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].grid()
    
    axs[1].plot(meta[:,1]*430/30) # Compensating for error when saving rewards
    axs[1].legend(["r"])
    axs[1].set(ylabel="Reward")
    axs[1].set_title("Average Validation Reward")
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].grid()

    pathToPlots = f"{os.getcwd()}/plots/"
    if not os.path.exists(os.path.dirname(pathToPlots)):
        os.makedirs(os.path.dirname(pathToPlots))
    # plt.show()
    plt.savefig(f"{pathToPlots}/success_reward{name}.png")

if __name__ == '__main__':
    #print(f"{int(sys.argv[1])}")
    visualizeTraining(sys.argv[1])
    # visualizeTraining(0, "")
    

