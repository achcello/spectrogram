import numpy as np
import matplotlib.pyplot as plt
gridRes=100001
class Ray(object):
    """
    ray
    """
    x = np.zeros(gridRes)
    z = np.zeros(gridRes)
    theta = np.zeros(gridRes)

    def __init__(self, x_0, z_0, theta_0):
        self.x[0] = x_0
        self.z[0] = z_0
        self.theta[0] = theta_0


def c(z):
    if z <= 0:
        z *= -1
        inner = 1 + (2 * g_0 * z) / (c_0 * (1 + beta))
        return c_0 * ((1 + beta) * np.sqrt(inner) - beta)
    else:
        return c_1


def cSimple(z):
    if z <= 0:
        return c_0
    else:
        return c_1


def g(z):
    return g_0 * (1 + beta) / ((c(z) / c_0) + beta)


def iterPos(ray, step):
        theta = ray.theta[step]
        x = ray.x[step]
        z = ray.z[step]
        if z > -400.001:
            if z > 0:
                dTheta = 0
            else:
                R_c = -1 / g(z) * c(ray.z[0]) / np.cos(ray.theta[0])
                dTheta = 2 * np.arctan(ds / 2 / R_c)
            ray.theta[step + 1] = theta + dTheta
            ray.x[step + 1] = x + ds * np.cos(theta)
            ray.z[step + 1] = z - ds * np.sin(theta)
        else:
            ray.theta[step + 1] = -1 * theta
            ray.x[step + 1] = x + ds * np.cos(theta)
            ray.z[step + 1] = -400


if __name__ == '__main__':
    beta = 0.86
    c_0 = 1400
    c_1 = 1500
    g_0 = 1.7
    gridRes = 100001
    for time in range(3000, 3002, 100):
        print('time:', time)
        endOfLength = time
        ds = endOfLength / gridRes
        s = np.linspace(0, endOfLength, gridRes)
        for angle in range(54, 57, 2):
            print('angle:', angle)
            ray = Ray(0, 500, np.deg2rad(angle))
            inv_snell_inv = c(ray.z[0]) / np.cos(ray.theta[0])
            # print(ray.x[0], ray.z[0], ray.theta[0], ray.v[0])
            for step in range(gridRes - 1):
                # print('iteration:', step)
                iterPos(ray, step)
                # print(ray.x[step], ray.z[step], ray.theta[step], ray.v[step])
            # print(ray.x[-1], ray.z[-1], np.rad2deg(ray.theta[-1]), ray.v[-1])
            plt.plot(ray.x, ray.z, 'b-')
            if angle == 30:
                plt.plot(ray.x, np.zeros(np.shape(ray.x)), 'c-')
                plt.plot(ray.x, -400 * np.ones(np.shape(ray.x)), 'k-')
        plt.ylim(-450, 900)
        plt.xlim(0, 2750)
        #plt.savefig(str(time) + '.png')
        plt.show()
