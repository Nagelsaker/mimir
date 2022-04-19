import numpy as np
import os, sys
def visualizeTraining(idx, name):
    import matplotlib.pyplot as plt
    path_to_metadata = f"{os.path.dirname(os.path.abspath(__file__))}/meta_data/success_reward{name}.npy"
    meta = np.load(path_to_metadata)
    plt.plot(meta[:,idx])
    plt.show()

if __name__ == '__main__':
    #print(f"{int(sys.argv[1])}")
    visualizeTraining(int(sys.argv[1]), sys.argv[2])
    

