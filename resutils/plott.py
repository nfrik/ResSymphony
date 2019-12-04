from mpl_toolkits.mplot3d import axes3d
import matplotlib
# matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import numpy as np
import json
import networkx as nx
from itertools import combinations

from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import scale, normalize, minmax_scale, StandardScaler
import pandas as pd

def plot3d(inmat,inputcirc=None,title=""):

    fig = plt.figure()

    ax = fig.add_subplot(235, projection='3d')
    std_scaler = StandardScaler()
    std_scaler.fit(inmat)
    inmat = std_scaler.transform(inmat)
    for cords in inmat:
        ax.scatter(cords[0], cords[1], cords[2], c='r' if cords[3] < 1. else 'b', marker='o' if cords[3] < 1. else '^')

    X = np.array(inmat)[:, :-1]
    y = np.array(inmat)[:, -1]

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
    colors = ['red', 'blue']
    for color, i in zip(colors, [min(y), max(y)]):
        ax2.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color)
        ax3.scatter(X_r[y == i, 1], X_r[y == i, 2], color=color)
        ax4.scatter(X_r[y == i, 2], X_r[y == i, 0], color=color)

    if inputcirc!=None:
        ax2 = fig.add_subplot(234)
        col_map, edgelist = json2edgelist(inputcirc)

        G, colors, edges = edgelist2graph(col_map, edgelist)

        pos = nx.spring_layout(G)

        nx.draw(G, with_labels=False, ax=ax2, edgelist=edges, pos=pos, edge_color=colors, node_size=10, linewidth=5.,
                font_size=8,title=title)


    plt.savefig("aggregated.png")
    plt.show()
    return ax

def pca_plotter(input,savepath=""):
    fig = plt.figure()

    std_scaler = StandardScaler()
    std_scaler.fit(input)
    input = std_scaler.transform(input)
    X = np.array(input)[:, :-1]
    y = np.array(input)[:, -1]

    # X = np.array(input)[:, :-1]
    # y = np.array(input)[:, -1]
    features=np.shape(X)[1]

    pca = PCA(n_components=features)
    X_r = pca.fit(X).transform(X)

    # ax2 = fig.add_subplot(231)
    # ax3 = fig.add_subplot(232)
    # ax4 = fig.add_subplot(233)
    # colors = ['red', 'blue']
    # for color, i in zip(colors, [min(y), max(y)]):
    #     ax2.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color)
    #     ax3.scatter(X_r[y == i, 1], X_r[y == i, 2], color=color)
    #     ax4.scatter(X_r[y == i, 2], X_r[y == i, 0], color=color)
    # pca = PCA(n_components=features)
    # X_r = pca.fit(X).transform(X)
    #
    combs=list(combinations(range(features),2))

    #create axes
    axs=[]
    pn=int(np.ceil(np.sqrt(features)))
    for comb,feat in zip(combs,range(features)):
        axs.append(fig.add_subplot(pn,pn,feat+1))

    colors = ['red', 'blue']
    for color, i in zip(colors, [min(y), max(y)]):
        for ax, comb in zip(axs, combs):
            ax.scatter(X_r[y == i, comb[0]], X_r[y == i, comb[1]], color=color)

    if savepath!="":
        plt.savefig(savepath)
    else:
        plt.show()


def plot_meas(datav=None, datai=None, title=''):
    plt.figure(figsize=(10, 5))
    #     df = pd.read_csv(fle)
    # plt.subplot(211)
    plt.plot(datav, np.multiply(1000, datai))
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (mA)')
    plt.title(title)
    plt.axhline(0, linewidth=.3, color='k')
    plt.axvline(0, linewidth=.3, color='k')
    # fig, ax = plt.subplots(212)
    # ax1, ax2 = two_scales(df['Time(s)']*1000,df['Drive(V)']*1000, df['Time(s)']*1000,df['Current(mA)']*50000000, 'r', 'b')

    fig, ax1 = plt.subplots()
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    ax1.plot(datav, 'b-')
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Voltage (V)', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    s2 = np.sin(2 * np.pi * t)
    ax2.plot(np.multiply(1000, datai), 'r-')
    ax2.set_ylabel('Current (mA)', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    # plt.show()

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


#Plots I vs V
def plot_meas(datav=None, datai=None, title='', tstep=0.0001):
    plt.figure(figsize=(10, 5))
    #     df = pd.read_csv(fle)
    # plt.subplot(211)
    plt.plot(datav, np.multiply(1000, datai))
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (mA)')
    plt.title(title)
    plt.axhline(0, linewidth=.3, color='k')
    plt.axvline(0, linewidth=.3, color='k')
    # fig, ax = plt.subplots(212)
    # ax1, ax2 = two_scales(df['Time(s)']*1000,df['Drive(V)']*1000, df['Time(s)']*1000,df['Current(mA)']*50000000, 'r', 'b')

    fig, ax1 = plt.subplots()
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    ax1.plot(np.arange(len(datav)) * tstep, datav, 'b-')
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Voltage (V)', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    s2 = np.sin(2 * np.pi * t)
    ax2.plot(np.arange(len(datav)) * tstep, np.multiply(1000, datai), 'r-')
    ax2.set_ylabel('Current (mA)', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    # plt.show()

    plt.show()

def two_scales(ax1, time, data1, data2, c1, c2):
    """

    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.plot(time, data1, color=c1)
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp')

    ax2.plot(time, data2, color=c2)
    ax2.set_ylabel('sin')
    return ax1, ax2


def plot_meas(fle='', title=''):
    if title == '':
        title = fle;
    plt.figure(figsize=(10, 5))
    df = pd.read_csv(fle)
    # plt.subplot(211)
    plt.plot(df['Drive(V)'], df['Current(mA)'] * 1000)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (uA)')
    plt.title(title)
    plt.axhline(0, linewidth=.3, color='k')
    plt.axvline(0, linewidth=.3, color='k')
    # fig, ax = plt.subplots(212)
    # ax1, ax2 = two_scales(df['Time(s)']*1000,df['Drive(V)']*1000, df['Time(s)']*1000,df['Current(mA)']*50000000, 'r', 'b')

    fig, ax1 = plt.subplots()
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    ax1.plot(df['Time(s)'], df['Drive(V)'], 'b.')
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Voltage (V)', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    s2 = np.sin(2 * np.pi * t)
    ax2.plot(df['Time(s)'], df['Current(mA)'] * 1000, 'r.')
    ax2.set_ylabel('Current (uA)', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    # plt.show()

    plt.show()


def plot_dual_meas(fle1='', fle2='', title=''):
    if title == '':
        title = fle1;
    df1 = pd.read_csv(fle1)
    tl = np.array(df1['Time(s)'])[-1]
    df2 = pd.read_csv(fle2)
    df2['Time(s)'] = df2['Time(s)'].add(tl)
    df = df1.append(df2)
    plt.figure(figsize=(8, 5))
    plt.plot(df['Drive(V)'], df['Current(mA)'] * 1000)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (uA)')
    plt.title(title)
    plt.axhline(0, linewidth=.3, color='k')
    plt.axvline(0, linewidth=.3, color='k')
    # fig, ax = plt.subplots(212)
    # ax1, ax2 = two_scales(df['Time(s)']*1000,df['Drive(V)']*1000, df['Time(s)']*1000,df['Current(mA)']*50000000, 'r', 'b')
    #     plt.show()
    #     plt.figure()
    fig, ax1 = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    color = 'tab:blue'
    ax1.plot(df['Time(s)'], df['Drive(V)'], color=color, linestyle=':', linewidth=2)
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Voltage (V)', color=color)
    ax1.tick_params('y', colors=color)

    color = 'tab:red'
    ax2 = ax1.twinx()
    s2 = np.sin(2 * np.pi * t)
    ax2.plot(df['Time(s)'], df['Current(mA)'] * 1000, color=color, linestyle='-', linewidth=2)
    ax2.set_ylabel('Current (uA)', color=color)
    ax2.tick_params('y', colors=color)

    fig.tight_layout()

    plt.show()


def plot_batch_meas(fles=[], title=''):
    if title == '':
        title = fles[0];

    df = pd.read_csv(fles[0])
    for i in range(len(fles) - 1):
        #         df1 = pd.read_csv(fles[i])
        tl = np.array(df['Time(s)'])[-1]
        df2 = pd.read_csv(fles[i + 1])
        df2['Time(s)'] = df2['Time(s)'].add(tl)
        df = df.append(df2)

    plt.figure(figsize=(6, 3))
    plt.plot(df['Drive(V)'], df['Current(mA)'] * 1000)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (uA)')
    plt.title(title)
    plt.axhline(0, linewidth=.3, color='k')
    plt.axvline(0, linewidth=.3, color='k')
    # fig, ax = plt.subplots(212)
    # ax1, ax2 = two_scales(df['Time(s)']*1000,df['Drive(V)']*1000, df['Time(s)']*1000,df['Current(mA)']*50000000, 'r', 'b')
    #     plt.show()
    #     plt.figure()
    fig, ax1 = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(6)
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    color = 'tab:blue'
    ax1.plot(df['Time(s)'], df['Drive(V)'], color=color, linestyle=':', linewidth=2)
    ax1.set_xlabel('time (s)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Voltage (V)', color=color)
    ax1.tick_params('y', colors=color)

    color = 'tab:red'
    ax2 = ax1.twinx()
    s2 = np.sin(2 * np.pi * t)
    ax2.plot(df['Time(s)'], df['Current(mA)'] * 1000, color=color, linestyle='-', linewidth=2)
    ax2.set_ylabel('Current (uA)', color=color)
    ax2.tick_params('y', colors=color)

    fig.tight_layout()

    plt.show()

def main():

    a = [[-0.0034, -0.0001, -0.0001, 0.],
         [-0.0001, -0., -0.0001, 1.],
         [-0.0033, -0.0001, -0.0001, 1.],
         [0., 0., 0.0001, 0.]]

    plot3d(a)

    with open(r'/home/nifrick/PycharmProjects/ResSymphony/results/n100_p0.045_k4_testxor_eqt0_5_date01-14-18-16_03_44_id35.json','r') as f:
        jdat=f.read()
        plot_json_graph(jdat)

if __name__ == "__main__":
    main()

