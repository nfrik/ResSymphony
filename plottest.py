# from __future__ import print_function
# import matplotlib
# matplotlib.use('QT5Agg')
# # matplotlib.use('PS')
# # matplotlib.use('WXAgg')
# import matplotlib.pyplot as plt
# import numpy as np
#
# print("Tensorflow Imported")
# plt.plot(np.arange(100))
# plt.show()


'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np


# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# n = 100
#
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()

a= [[-0.0034, -0.0001, -0.0001,  0.    ],
    [-0.0001, -0.,     -0.0001,  1.    ],
    [-0.0033, -0.0001, -0.0001,  1.    ],
    [ 0.,      0.,      0.0001,  0.    ]]


def plot3d(input):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cords in input:
        ax.scatter(cords[0],cords[1],cords[2],c='r' if cords[3]<1. else 'b', marker='o' if cords[3]<1. else '^')

    plt.show()

plot3d(a)