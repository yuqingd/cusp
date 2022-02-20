import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot(goals, filename, contour=False, perturb=0., step=0):
    fig = plt.figure()
    goals = np.asarray(np.squeeze(goals))

    if goals.shape[-1] == 3:
        goals = goals[..., :2]
    elif goals.shape[-1] > 2:
        # reshape for subgoals, TODO: remove hard coding
        goals = goals.reshape(-1, 2)

    if len(goals.shape) == 2:
        xs = np.squeeze(goals[..., 0])
        ys = np.squeeze(goals[..., 1])
    else:
        xs = goals
        ys = np.zeros_like(xs)
    colours = cm.rainbow(np.linspace(0, 1, len(ys)))

    for x, y, c in zip(xs, ys, colours):
        plt.scatter(x, y, color=c, marker='.')

    if contour:
        center = 0.1
        x = np.linspace(-.3, .3)
        y = np.linspace(-.3, .3)

        X, Y = np.meshgrid(x, y)
        # Z = (X-.2+perturb) ** 2 + (Y-.2+perturb) ** 2
        Z = (X+center-perturb) ** 2 + (Y-center+perturb) ** 2 
        Z = np.clip(Z, a_min=None, a_max=.01)

        Z_base = (X+center) ** 2 + (Y-center) ** 2 
        Z_base = np.clip(Z_base, a_min=None, a_max=.01)

        plt.contour(X, Y, -Z,  10, cmap='RdGy')
        if perturb>0:
            plt.contour(X, Y, -Z_base,  10, color='blue')

    plt.savefig(filename)
    plt.close(fig)

