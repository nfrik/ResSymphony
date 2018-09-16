from mpl_toolkits.mplot3d import axes3d
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# plt.switch_backend('agg')
import numpy as np
import json
import networkx as nx

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import scale, normalize, minmax_scale, StandardScaler

def plot3d(input,inputcirc=None,title=""):

    fig = plt.figure()

    ax = fig.add_subplot(235, projection='3d')
    std_scaler = StandardScaler()
    std_scaler.fit(input)
    input = std_scaler.transform(input)
    for cords in input:
        ax.scatter(cords[0], cords[1], cords[2], c='r' if cords[3] < 1. else 'b', marker='o' if cords[3] < 1. else '^')

    X = np.array(input)[:, :-1]
    y = np.array(input)[:, -1]

    logreg = linear_model.LogisticRegression(C=300.5, verbose=True, tol=1e-8, fit_intercept=True)
    logreg.fit(X, y)

    zlr = lambda x, y: (-logreg.intercept_[0] - logreg.coef_[0][0] * x - logreg.coef_[0][1] * y) / logreg.coef_[0][2]

    tmp = np.linspace(-1, 1, 10)
    xlg, ylg = np.meshgrid(tmp, tmp)
    ax.plot_surface(xlg, ylg, zlr(xlg, ylg),color='yellow')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    var = logreg.score(X,y)
    ax.set_title("Score: {}".format(str(var)))

    pca = PCA(n_components=3)
    X_r = pca.fit(X).transform(X)
    print(X_r)

    ax2 = fig.add_subplot(231)
    ax3 = fig.add_subplot(232)
    ax4 = fig.add_subplot(233)
    x0, x1, x2 = X_r[:, 0].reshape((-1,1)), X_r[:, 1], X_r[:, 2]
    ax2.scatter(x0, x1, c=y, cmap="RdBu", vmin=-0.2, vmax=1.2, edgecolor='white', linewidth=1)
    ax3.scatter(x1, x2, c=y, cmap="RdBu", vmin=-0.2, vmax=1.2, edgecolor='white', linewidth=1)
    ax4.scatter(x2, x0, c=y, cmap="RdBu", vmin=-0.2, vmax=1.2, edgecolor='white', linewidth=1)

    # colors = ['red', 'blue']
    # for color, i in zip(colors, [min(y), max(y)]):
    #     x0,x1,x2=X_r[y == i, 0], X_r[y == i, 1],X_r[y == i, 2]
    #     ax2.scatter(x0, x1, color=color)
    #     ax3.scatter(x1, x2, color=color)
    #     ax4.scatter(x2, x0, color=color)

    #intercept
    # x0,x1,x2=X_r[:,0],X_r[:,1],X_r[:,2]
    x = np.arange(-1, 1, 0.1)
    x0 = x0.reshape((-1, 1))
    x1 = x1.reshape((-1, 1))
    x2 = x2.reshape((-1, 1))
    logreg01 = linear_model.LogisticRegression(C=30000.5, verbose=True, tol=1e-11, fit_intercept=True)
    logreg12 = linear_model.LogisticRegression(C=30000.5, verbose=True, tol=1e-11, fit_intercept=True)
    logreg20 = linear_model.LogisticRegression(C=30000.5, verbose=True, tol=1e-11, fit_intercept=True)
    logreg01.fit(np.hstack((x0,x1)), y)
    logreg12.fit(np.hstack((x1,x2)), y)
    logreg20.fit(np.hstack((x2,x0)), y)

    xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs01 = logreg01.predict_proba(grid)[:, 1].reshape(xx.shape)
    contour = ax2.contourf(xx, yy, probs01, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    probs12 = logreg12.predict_proba(grid)[:, 1].reshape(xx.shape)
    contour = ax3.contourf(xx, yy, probs12, 25, cmap="RdBu",
                           vmin=0, vmax=1)
    probs20 = logreg20.predict_proba(grid)[:, 1].reshape(xx.shape)
    contour = ax4.contourf(xx, yy, probs20, 25, cmap="RdBu",
                           vmin=0, vmax=1)

    # ax_c = f.colorbar(contour)
    # ax_c.set_label("$P(y = 1)$")
    # ax_c.set_ticks([0, .25, .5, .75, 1])

    im1=ax2.scatter(x0, x1, c=y.reshape((-1,1)), s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    axc1=fig.colorbar(im1, cax=cax, orientation='vertical')
    axc1.set_label("$P(y = 1)$")
    axc1.set_ticks([0, .25, .5, .75, 1])

    im2=ax3.scatter(x1, x2, c=y.reshape((-1, 1)), s=50,
                cmap="RdBu", vmin=-.2, vmax=1.2,
                edgecolor="white", linewidth=1)

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    axc2 = fig.colorbar(im2, cax=cax, orientation='vertical')
    axc2.set_label("$P(y = 1)$")
    axc2.set_ticks([0, .25, .5, .75, 1])

    im3=ax4.scatter(x2, x0, c=y.reshape((-1, 1)), s=50,
                cmap="RdBu", vmin=-.2, vmax=1.2,
                edgecolor="white", linewidth=1)
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    axc4 = fig.colorbar(im3, cax=cax, orientation='vertical')
    axc4.set_label("$P(y = 1)$")
    axc4.set_ticks([0, .25, .5, .75, 1])

    ax2.set(aspect="equal",
           xlim=(-2, 2), ylim=(-2, 2),
           xlabel="$X_0$", ylabel="$X_1$")
    ax3.set(aspect="equal",
           xlim=(-2, 2), ylim=(-2, 2),
           xlabel="$X_1$", ylabel="$X_2$")
    ax4.set(aspect="equal",
           xlim=(-2, 2), ylim=(-2, 2),
           xlabel="$X_2$", ylabel="$X_0$")

    # y01 = -(logreg01.intercept_[0] + x * logreg01.coef_[0][0]) / logreg01.coef_[0][1]
    # y12 = -(logreg12.intercept_[0] + x * logreg12.coef_[0][0]) / logreg12.coef_[0][1]
    # y20 = -(logreg20.intercept_[0] + x * logreg20.coef_[0][0]) / logreg20.coef_[0][1]
    # ax2.plot(x, y01)
    # ax3.plot(x, y12)
    # ax4.plot(x, y20)

    if inputcirc!=None:
        ax2 = fig.add_subplot(234)
        col_map, edgelist = json2edgelist(inputcirc)

        G, colors, edges = edgelist2graph(col_map, edgelist)

        pos = nx.spring_layout(G)

        nx.draw(G, with_labels=False, ax=ax2, edgelist=edges, pos=pos, edge_color=colors, node_size=10, linewidth=5.,
                font_size=8,title=title)


    plt.savefig("aggregated.png")
    plt.show()


def plot_json_graph(dictdata,imagepath=""):

    col_map, edgelist = json2edgelist(dictdata)

    G, colors, edges = edgelist2graph(col_map, edgelist)

    pos = nx.spring_layout(G)

    nx.draw(G, with_labels=True,edgelist=edges,pos=pos,edge_color=colors,node_size=10,linewidth=5.,font_size=8)

    if imagepath!="":
        plt.savefig(imagepath)
    else:
        plt.show()


def edgelist2graph(col_map, edgelist):
    G = nx.Graph()
    for e in edgelist:
        G.add_edges_from([e[1:]], color=col_map.get(e[0], 'cyan'))
    edges, colors = zip(*nx.get_edge_attributes(G, 'color').items())
    return G, colors, edges


def json2edgelist(dictdata):
    jsondata = json.loads(dictdata)
    edgelist = []
    for k in jsondata.keys():
        if k != '0':
            edgelist.append(jsondata[k][0:3])
    col_map = {'m': 'blue',  # internal
               'g': 'green',  # output
               'r': 'green',  # output
               'R': 'red'}  # input
    return col_map, edgelist


def main():

    a = [[-0.0034, -0.0001, -0.0001, 0.],
         [-0.0001, -0., -0.0001, 1.],
         [-0.0033, -0.0001, -0.0001, 1.],
         [0., 0., 0.0001, 0.]]

    plot3d(a)

    with open(r'/home/nifrick/PycharmProjects/ressymphony/results/n100_p0.045_k4_testxor_eqt0_5_date01-14-18-16_03_44_id35.json','r') as f:
        jdat=f.read()
        plot_json_graph(jdat)

if __name__ == "__main__":
    main()

