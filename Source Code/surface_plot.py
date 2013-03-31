import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

def plot():
    """
    Plot 2-D array in xyz space using values and 2-D grid
    Output:
      3-D plot
    """
    fig = plt.figure()
    ax = Axes3D(fig)
    f = open('DEM.asc', 'r')
    Z = np.loadtxt(f,unpack = True)
    f.close()
    X, Y = np.mgrid[:Z.shape[0],:Z.shape[1]]
    ax.plot_surface(X,Y,Z,rstride=2,cstride=2,cmap="gray",linewidth=0,antialiased=False)

    ax.set_xlabel('X dirn')
    ax.set_ylabel('Y dirn')
    ax.set_zlabel('Elevation')
    plt.show()


if __name__ == "__main__":
    plot()
