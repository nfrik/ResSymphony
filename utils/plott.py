from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import networkx as nx

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import scale, normalize, minmax_scale, StandardScaler

def plot3d(input,inputcirc=None):


    fig = plt.figure()

    ax = fig.add_subplot(235, projection='3d')
    std_scaler = StandardScaler()
    std_scaler.fit(input)
    input = std_scaler.transform(input)
    for cords in input:
        ax.scatter(cords[0], cords[1], cords[2], c='r' if cords[3] < 1. else 'b', marker='o' if cords[3] < 1. else '^')

    X = np.array(input)[:, :3]
    y = np.array(input)[:, 3]

    logreg = linear_model.LogisticRegression(C=300.5, verbose=True, tol=1e-8, fit_intercept=True)
    logreg.fit(X, y)

    zlr = lambda x, y: (-logreg.intercept_[0] - logreg.coef_[0][0] * x - logreg.coef_[0][1] * y) / logreg.coef_[0][2]

    tmp = np.linspace(-1, 1, 10)
    xlg, ylg = np.meshgrid(tmp, tmp)
    ax.plot_surface(xlg, ylg, zlr(xlg, ylg))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    var = np.var(logreg.predict(X) - y)
    ax.set_title("LogRegVar: {}".format(str(var)))

    if inputcirc!=None:
        ax2 = fig.add_subplot(234)
        col_map, edgelist = json2edgelist(inputcirc)

        G, colors, edges = edgelist2graph(col_map, edgelist)

        pos = nx.spring_layout(G)

        nx.draw(G, with_labels=True, ax=ax2, edgelist=edges, pos=pos, edge_color=colors, node_size=10, linewidth=5.,
                font_size=8)

        pca = PCA(n_components=3)
        X_r = pca.fit(X).transform(X)

        ax2 = fig.add_subplot(231)
        ax3 = fig.add_subplot(232)
        ax4 = fig.add_subplot(233)
        colors = ['red', 'blue']
        for color, i in zip(colors, [-1., 1.]):
            ax2.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color)
            ax3.scatter(X_r[y == i, 1], X_r[y == i, 2], color=color)
            ax4.scatter(X_r[y == i, 2], X_r[y == i, 0], color=color)



    plt.show()


def plot_json_graph(dictdata):

    col_map, edgelist = json2edgelist(dictdata)

    G, colors, edges = edgelist2graph(col_map, edgelist)

    pos = nx.spring_layout(G)

    nx.draw(G, with_labels=True,edgelist=edges,pos=pos,edge_color=colors,node_size=10,linewidth=5.,font_size=8)
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

    # a = [[-0.0034, -0.0001, -0.0001, 0.],
    #      [-0.0001, -0., -0.0001, 1.],
    #      [-0.0033, -0.0001, -0.0001, 1.],
    #      [0., 0., 0.0001, 0.]]
    #
    # plot3d(a)

    with open(r'/home/nifrick/PycharmProjects/ResSymphony/results/n100_p0.045_k4_testxor_eqt0_5_date01-14-18-16_03_44_id35.json','r') as f:
        jdat=f.read()
        plot_json_graph(jdat)

if __name__ == "__main__":
    main()