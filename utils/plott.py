from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np


from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.preprocessing import scale, normalize, minmax_scale, StandardScaler


def plot3d(input):


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    std_scaler = StandardScaler()
    std_scaler.fit(input)
    input=std_scaler.transform(input)
    for cords in input:
        ax.scatter(cords[0],cords[1],cords[2],c='r' if cords[3]<1. else 'b', marker='o' if cords[3]<1. else '^')



    X=np.array(input)[:, :3]
    y=np.array(input)[:, 3]
    # svc = SVC(kernel='linear',verbose=True)
    # svc.fit(np.array(input)[:,:3],np.array(input)[:,3])
    logreg = linear_model.LogisticRegression(C=300.5,verbose=True,tol=1e-8,fit_intercept=True)
    sgd = linear_model.SGDClassifier(max_iter=1000,tol=1e-10,verbose=True,alpha=1e-2)
    logreg.fit(X, y)
    sgd.fit(X,y)

    # z = lambda x, y: (-svc.intercept_[0] - svc.coef_[0][0] * x - svc.coef_[0][1]) / svc.coef_[0][2]
    zlr = lambda x, y: (-logreg.intercept_[0] - logreg.coef_[0][0] * x - logreg.coef_[0][1] * y) / logreg.coef_[0][2]
    # sgdz = lambda x, y: (-sgd.intercept_[0] - sgd.coef_[0][0] * x - sgd.coef_[0][1] * y) / sgd.coef_[0][2]

    tmp = np.linspace(-1, 1, 10)
    x, y = np.meshgrid(tmp, tmp)
    ax.plot_surface(x, y, zlr(x, y))
    # ax.plot_surface(x, y, sgdz(x, y))
    # ax.plot_surface(x, y, z(x, y))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

def main():

    a = [[-0.0034, -0.0001, -0.0001, 0.],
         [-0.0001, -0., -0.0001, 1.],
         [-0.0033, -0.0001, -0.0001, 1.],
         [0., 0., 0.0001, 0.]]

    plot3d(a)

if __name__ == "__main__":
    main()