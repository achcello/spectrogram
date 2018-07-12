import numpy as np
import matplotlib.pyplot as plt


class Ray(object):
    """
    ray
    """
    gridRes = 100001
    x = np.zeros(gridRes)
    z = np.zeros(gridRes)
    theta = np.zeros(gridRes)
    v = np.zeros(gridRes)

    def __init__(self, x_0, z_0, theta_0):
        self.x[0] = x_0
        self.z[0] = z_0
        self.theta[0] = theta_0
        self.v[0] = cSimple(z_0)


def cSimple(z):
    if z <= 0:
        z *= -1
        inner = 1 + (2 * g_0 * z) / (c_0 * (1 + beta))
        return c_0 * ((1 + beta) * np.sqrt(inner) - beta)
    else:
        return c_1


def cSimpleNot(z):
    if z <= 0:
        return c_0
    else:
        return c_1


def iterPos(ray, step):
    x_0 = ray.x[step - 1]
    z_0 = ray.z[step - 1]
    theta_0 = ray.theta[step - 1]
    v_0 = ray.v[step - 1]
    ray.x[step] = x_0 + v_0 * dt * np.sin(theta_0)
    ray.z[step] = z_0 - v_0 * dt * np.cos(theta_0)
    ray.v[step] = cSimple(ray.z[step])
    # if not np.isclose(ray.v[step], ray.v[step - 1]):
        # print('Change!', ray.v[step] / ray.v[step - 1])
    ray.theta[step] = np.arcsin((ray.v[step] / ray.v[step - 1]) * np.sin(ray.theta[step - 1]))
    #if np.isclose(ray.theta[step], ray.theta[step - 1]) and ray.z[step] < 0:
    #    ray.theta[step] += 0.00001


if __name__ == '__main__':
    beta = 0.86
    c_0 = 1400
    c_1 = 1500
    g_0 = 1.7
    gridRes = 100001
    endOfTime = 4.25
    dt = endOfTime / gridRes
    t = np.linspace(0, endOfTime, gridRes)
    for angle in range(30, 31):
        print('angle:', angle)
        ray = Ray(0, 500, np.deg2rad(angle))
        #print(ray.x[0], ray.z[0], ray.theta[0], ray.v[0])
        for step in range(1, gridRes):
            # print('iteration:', step)
            iterPos(ray, step)
            # print(ray.x[step], ray.z[step], ray.theta[step], ray.v[step])
        #print(ray.x[-1], ray.z[-1], np.rad2deg(ray.theta[-1]), ray.v[-1])
        plt.plot(ray.x, ray.z, 'b-')
        if angle == 30:
            plt.plot(ray.x, np.zeros(np.shape(ray.x)), 'k-')
        print(np.rad2deg(ray.theta[:14]))
    plt.show()
