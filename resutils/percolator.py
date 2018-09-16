import requests
import json
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import networkx as nx
import resutils.nxgtutils as nxutils
import numpy as np
import graph_tool.all as gt
from matplotlib import cm
import itertools
from statsmodels.graphics.mosaicplot import mosaic
from tqdm import tqdm
from resutils.graph2json import *
from resutils import utilities
from resutils import netfitter2 as netfitter
from scipy import signal

import requests
import json
import time


def load_graph_from_json_file(
        path='/home/nifrick/Documents/development/jupyter/networktest/spatial_percolation_data/16box_5.json'):
    with open(path) as f:
        data = json.load(f)
    graph = load_graph_from_json(data)
    return graph


def load_graph_from_json(data):
    G = nx.Graph()
    for k in data['pos'].keys():
        G.add_node(int(k), pos=(data['pos'][k][0], data['pos'][k][1]),
                   pos3d=(data['pos'][k][0], data['pos'][k][1], data['pos'][k][2]))

    for k in data['list']:
        for n in data['list'][k]:
            G.add_edge(int(k), int(n[1]), edgetype='m', edgeclass=n[0])

    pos = nx.get_node_attributes(G, 'pos')
    pos3d = nx.get_node_attributes(G, 'pos3d')

    return G


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_simple_paths_graph(G, start, end, cutoff=None):
    paths = nx.all_simple_paths(G, start, end, cutoff=cutoff)
    sG = nx.Graph()

    pos = nx.get_node_attributes(G, 'pos')
    pos3d = nx.get_node_attributes(G, 'pos3d')

    for path in paths:
        for pair in pairwise(path):
            sG.add_edge(pair[0], pair[1])

    nx.set_node_attributes(sG, pos, 'pos')
    nx.set_node_attributes(sG, pos3d, 'pos3d')
    return sG


def get_simple_paths_graph_gtalg(G, start, end, cutoff=None):
    pos = nx.get_node_attributes(G, 'pos')
    pos3d = nx.get_node_attributes(G, 'pos3d')
    gtG = nxutils.nx2gt(nxG=G)
    paths = gt.all_paths(gtG, gt.find_vertex(gtG, gtG.vertex_properties['id'], start)[0],
                         gt.find_vertex(gtG, gtG.vertex_properties['id'], end)[0], cutoff=cutoff)
    #     paths=gt.all_shortest_paths(gtG,gt.find_vertex(gtG,gtG.vertex_properties['id'],start)[0],gt.find_vertex(gtG,gtG.vertex_properties['id'],end)[0])

    sG = nx.Graph()

    for path in paths:
        for pair in pairwise(path):
            #             sG.add_edge(pair[0],pair[1])
            sG.add_edge(int(gtG.vertex_properties['id'][pair[0]]), int(gtG.vertex_properties['id'][pair[1]]))

    nx.set_node_attributes(sG, values=pos, name='pos')
    nx.set_node_attributes(sG, values=pos3d, name='pos3d')
    return sG


def wire_nodes_on_electrodes(subgraph, electrodeslist=[]):
    subgraph = subgraph.copy()
    for electrodearray in electrodeslist:
        for electrode in electrodearray.values():
            #             for node in electrodearray[electrode]:
            for wire in pairwise(electrode):  # itertools.combinations(electrode,2):
                subgraph.add_edge(wire[0], wire[1], edgetype='w')
    return subgraph


def convert_devices_to_resistors(graph, min_length=10, max_length=1e8, out_range_dev='m', in_range_dev='r'):
    #     graph_copy=nx.Graph()
    #     graph_copy.add_nodes_from(graph.nodes(data=True))
    #     graph_copy.add_edges_from(s_graph.edges(data=True))
    graph_copy = graph.copy()
    pos3d = nx.get_node_attributes(graph_copy, 'pos3d')
    edges = graph_copy.edges()
    for e in edges:
        p1 = np.array(pos3d[e[0]])
        p2 = np.array(pos3d[e[1]])
        dist = np.linalg.norm(p1 - p2)
        if dist >= min_length and dist < max_length:
            graph_copy[e[0]][e[1]]['edgetype'] = in_range_dev
        else:
            graph_copy[e[0]][e[1]]['edgetype'] = out_range_dev
    return graph_copy


def convert_edgeclass_to_device(graph, mem='wo3', res='ag'):
    graph_copy = graph.copy()
    edges = graph_copy.edges()
    for e, v in nx.get_edge_attributes(graph_copy, 'edgeclass').items():
        if mem.lower() in v.lower():
            graph_copy[e[0]][e[1]]['edgetype'] = 'm'
        elif res.lower() in v.lower():
            graph_copy[e[0]][e[1]]['edgetype'] = 'r'
    return graph_copy


def prune_dead_edges(graph, runs=25):
    G = graph.copy()
    nbs = []
    for run in range(runs):
        for n in sorted([z for z in sorted(G.degree, key=lambda x: x[1], reverse=False) if z[1] == 1]):
            nbs.append([n[0], list(G.neighbors(n[0]))[0]])

        for nb in nbs:
            try:
                G.remove_edge(nb[0], nb[1])
            except:
                pass
        # remove isolates
        G.remove_nodes_from(list(nx.isolates(G)))
    return G


def plot_nxgraph(G, pos=None, edge_colors=None):
    plt.figure(figsize=(7, 7))
    if pos == None and edge_colors == None:
        #         nx.draw_networkx(G,node_size=50,font_size=9)
        nx.draw_networkx(G, node_size=50, font_size=9)
    elif pos != None and edge_colors == None:
        #         nx.draw_networkx(G,pos,node_size=50,font_size=9)
        nx.draw_networkx(G, pos=pos, node_size=50, font_size=9)
    elif pos == None and edge_colors != None:
        nx.draw_networkx(G, node_size=50, font_size=9, edge_color=edge_colors)
    elif pos != None and edge_colors != None:
        nx.draw_networkx(G, pos=pos, node_size=50, font_size=9, edge_color=edge_colors)
    else:
        nx.draw_networkx(G, node_size=50, font_size=9)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()


def plot_nxdegree_hist(G, n=10):
    plt.figure(figsize=(5, 5))
    plt.hist(list(dict(nx.degree(G)).values()), n, density=True)
    plt.title("Degree Histogram")
    plt.ylabel("Frequency")
    plt.xlabel("Degree")
    plt.show()


def plot_nxdegree_log(G, ax=None):
    degree_sequence = sorted(dict(nx.degree(G)).values(), reverse=True)  # degree sequence
    # print "Degree sequence", degree_sequence
    dmax = max(degree_sequence)
    plt.figure()
    plt.loglog(degree_sequence, 'b-', marker='o')
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")
    plt.show()


def plot_device_length_histogram(graph, attribute='', attribute_value=''):
    pos3d = nx.get_node_attributes(graph, 'pos3d')
    if attribute != '':
        edges = {k: v for k, v in nx.get_edge_attributes(graph, attribute).items() if v == attribute_value}
    else:
        edges = graph.edges()
    contact_lengths = []
    for e in edges:
        p1 = np.array(pos3d[e[0]])
        p2 = np.array(pos3d[e[1]])
        dist = np.linalg.norm(p1 - p2)
        contact_lengths.append(dist)

    plt.figure()
    plt.hist(contact_lengths, bins=100)
    plt.xlabel("device length")
    plt.ylabel("nubmer of devices")
    plt.show()


# plot_nxgraph(G,pos)

# ax = plt.gca(projection='3d')


# find subraphs which have connections between xmin+/-delta and xmax+/-delta

# subgraphs=list(nx.connected_component_subgraphs(G))
# accepted_graphs=[]
# xdelta=25
# xmin=min([k[0] for k in pos.values()])
# xmax=max([k[0] for k in pos.values()])

def get_nodes_within_xboundary(pos, lowerb, upperb):
    nodes = []
    if lowerb > upperb:
        lowerb, upperb = upperb, lowerb

    for k, v in zip(pos.keys(), pos.values()):
        if v[0] >= lowerb and v[0] <= upperb:
            nodes.append(k)
    return nodes


def get_nodes_within_3dboundary(pos, lowerbx, upperbx, lowerby, upperby, lowerbz, upperbz):
    nodes = []
    if lowerbx > upperbx:
        lowerbx, upperbx = upperbx, lowerbx
    if lowerby > upperby:
        lowerby, upperby = upperby, lowerby
    if lowerbz > upperbz:
        lowerbz, upperbz = upperbz, lowerbz

    for k, v in zip(pos.keys(), pos.values()):
        if v[0] >= lowerbx and v[0] <= upperbx and v[1] >= lowerby and v[1] <= upperby and v[2] >= lowerbz and v[
            2] <= upperbz:
            nodes.append(k)
    return nodes


def get_nodes_for_electrode(pos, elects, xmin, xmax, ymax, zmax):
    elects_bucket = {}
    for k in elects.keys():
        y1, z1, h, w = elects[k][0], elects[k][1], elects[k][2], elects[k][3]
        y2 = y1 + h
        z2 = z1 + w
        y1 = y1 * ymax
        y2 = y2 * ymax
        z1 = z1 * zmax
        z2 = z2 * zmax
        elects_bucket[k] = get_nodes_within_3dboundary(pos, lowerbx=xmin, upperbx=xmax, lowerby=y1, upperby=y2,
                                                       lowerbz=z1, upperbz=z2)
    return elects_bucket


def get_pos_for_subgraph(subgraph, superpos):
    subpos = {}
    for e in subgraph.nodes():
        if e in superpos.keys():
            subpos[e] = superpos[e]
    return subpos


# get_pos_for_subgraph(subgraphs[7],pos)

def get_connected_graphs(supergraph, xdelta):
    pos = nx.get_node_attributes(supergraph, 'pos')
    subgraphs = list(nx.connected_component_subgraphs(supergraph))
    xmin = min([k[0] for k in pos.values()])
    xmax = max([k[0] for k in pos.values()])
    accepted_graphs = []
    for sg in subgraphs:
        subpos = get_pos_for_subgraph(sg, pos)
        nodesright = get_nodes_within_xboundary(subpos, xmax - xdelta, 1e8)
        nodesleft = get_nodes_within_xboundary(subpos, -1e8, xmin + xdelta)
        if len(nodesright) > 0 and len(nodesleft) > 0:
            accepted_graphs.append(sg)
    return accepted_graphs


def get_3d_minmax(supergraph):
    pos3d = nx.get_node_attributes(supergraph, 'pos3d')
    xmin = min([k[0] for k in pos3d.values()])
    xmax = max([k[0] for k in pos3d.values()])
    ymin = min([k[1] for k in pos3d.values()])
    ymax = max([k[1] for k in pos3d.values()])
    zmin = min([k[2] for k in pos3d.values()])
    zmax = max([k[2] for k in pos3d.values()])
    return xmin, xmax, ymin, ymax, zmin, zmax


# retrieves graphs connected to elects1 and elects2 - electrode arrays along x axis
def get_connected_graphs_electrodes(supergraph, elects1, elects2, delta, box, boy, boz):
    pos3d = nx.get_node_attributes(supergraph, 'pos3d')
    subgraphs = list(nx.connected_component_subgraphs(supergraph))
    #     xmin,xmax,ymin,ymax,zmin,zmax = get_3d_minmax(supergraph)
    accepted_graphs = []
    for sg in subgraphs:
        subpos = get_pos_for_subgraph(sg, pos3d)

        elects1_bucket = get_nodes_for_electrode(subpos, elects1, 0 - delta, 0 + delta, boy, boz)
        elects2_bucket = get_nodes_for_electrode(subpos, elects2, box - delta, boy + delta, boy, boz)
        for k in elects1_bucket.keys():
            elects1_bucket[k] = len(elects1_bucket[k])
        for k in elects2_bucket.keys():
            elects2_bucket[k] = len(elects2_bucket[k])

        #         elects1_bucket={}
        #         for k in elects1:
        #             y1,z1,h,w=elects1[k][0],elects1[k][1],elects1[k][2],elects1[k][3]
        #             y2=y1+h
        #             z2=z1+w
        #             y1=y1*ymax
        #             y2=y2*ymax
        #             z1=z1*zmax
        #             z2=z2*zmax
        #             elects1_bucket[k]=len(get_nodes_within_3dboundary(subpos,lowerbx=-1e8,upperbx=xmin+delta,lowerby=y1,upperby=y2,lowerbz=z1,upperbz=z2))

        #         elects2_bucket={}
        #         for k in elects2:
        # #             y1,z1,y2,z2=elects2[k][0]*ymax,elects2[k][1]*zmax,elects2[k][2]*ymax,elects2[k][3]*zmax
        #             y1,z1,h,w=elects2[k][0],elects2[k][1],elects2[k][2],elects2[k][3]
        #             y2=y1+h
        #             z2=z1+w
        #             y1=y1*ymax
        #             y2=y2*ymax
        #             z1=z1*zmax
        #             z2=z2*zmax
        #             elects2_bucket[k]=len(get_nodes_within_3dboundary(subpos,lowerbx=xmax-delta,upperbx=1e8,lowerby=y1,upperby=y2,lowerbz=z1,upperbz=z2))

        if 0 not in list(elects1_bucket.values()) and 0 not in list(elects2_bucket.values()):
            accepted_graphs.append(sg)
    return accepted_graphs


def plot_3d_scatter(xs, ys, zs, xslabel='x', yslabel='y', zslabel='z', ax=None, colorsMap='jet', title=''):
    cs = [1, 1]
    if ax == None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.scatter(xs, ys, zs, c='r', s=30)
    # ax.plot(xs,ys,zs, color='r')

    p3d = ax.scatter(xs, ys, zs, s=len(xs), c=zs, cmap=cm.coolwarm)
    fig.colorbar(p3d)
    plt.title(title)
    ax.set_xlabel(xslabel)
    ax.set_ylabel(yslabel)
    ax.set_zlabel(zslabel)
    ax.view_init(elev=20, azim=90)
    plt.show()
    return ax


def plot_pos3d(accepted_graph, ax=None, title=''):
    pos3d = nx.get_node_attributes(accepted_graph, 'pos3d')
    xs = [pos3d[k][0] for k in accepted_graph.nodes()]
    ys = [pos3d[k][1] for k in accepted_graph.nodes()]
    zs = [pos3d[k][2] for k in accepted_graph.nodes()]

    if ax == None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.scatter(xs, ys, zs, c='r', s=5)
    # ax.plot(xs,ys,zs, color='r')
    for e in accepted_graph.edges():
        x1 = pos3d[e[0]][0]
        y1 = pos3d[e[0]][1]
        z1 = pos3d[e[0]][2]
        x2 = pos3d[e[1]][0]
        y2 = pos3d[e[1]][1]
        z2 = pos3d[e[1]][2]
        x = [x1, x2]
        y = [y1, y2]
        z = [z1, z2]
        edgetype = {}
        try:
            edgetype = accepted_graph[e[0]][e[1]]['edgetype']
            if 'm' in edgetype:
                ax.plot(x, y, z, c='b')
            else:
                ax.plot(x, y, z, c='k')
        except:
            ax.plot(x, y, z, c='k')
            pass

    plt.title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=90)
    plt.show()
    return ax


def plot_electrodes(ax=None, els=None, xmax=1, ymax=1, zmax=1, xdelta=1):
    x1, x2 = xmax - xdelta, xmax + xdelta
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * 10
    if ax == None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    for k, n in zip(list(els.keys()), range(len(list(els.keys())))):
        #     y1,z1,y2,z2=els[k][0],els[k][1],els[k][2],els[k][3]
        y1, z1, h, w = els[k][0], els[k][1], els[k][2], els[k][3]
        y2 = y1 + h
        z2 = z1 + w
        y1 = y1 * ymax
        y2 = y2 * ymax
        z1 = z1 * zmax
        z2 = z2 * zmax

        edges = []
        edges.append([[y1, z1, x1], [y1, z2, x1]])
        edges.append([[y1, z2, x1], [y2, z2, x1]])
        edges.append([[y2, z2, x1], [y2, z1, x1]])
        edges.append([[y2, z1, x1], [y1, z1, x1]])

        edges.append([[y1, z1, x2], [y1, z2, x2]])
        edges.append([[y1, z2, x2], [y2, z2, x2]])
        edges.append([[y2, z2, x2], [y2, z1, x2]])
        edges.append([[y2, z1, x2], [y1, z1, x2]])

        edges.append([[y1, z1, x1], [y1, z1, x2]])
        edges.append([[y1, z2, x1], [y1, z2, x2]])
        edges.append([[y2, z2, x1], [y2, z2, x2]])
        edges.append([[y2, z1, x1], [y2, z1, x2]])

        #     edges.append([y1,y1,y2,y2])
        #     edges.append([z1,z2,z1,z2])
        #     edges.append([x1,x1,x1,x1])
        for e in edges:
            y = np.array(e)[:, 0].tolist()
            z = np.array(e)[:, 1].tolist()
            x = np.array(e)[:, 2].tolist()
            ax.plot(x, y, z, colors[n])
    return ax


def get_electrodes_rects(els=[1, 1], gap=0.2):
    list1 = list(range(els[0]))
    list2 = list(range(els[1]))
    data = {}
    for k in list(itertools.product(list1, list2)):
        data[k] = 1
    ax = plt.axes()
    m = mosaic(data, ax=ax, gap=gap, title='complete dictionary')
    plt.close()
    ret = {}
    for k, i in zip(m[1].keys(), range(len(m[1].keys()))):
        ret[i] = m[1][k]
    return ret


class Percolator:
    serverUrl = ""

    def __init__(self, serverUrl):
        Percolator.serverUrl = serverUrl

    def generate_network_native(self, data):
        url = Percolator.serverUrl + 'generate'
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        data = json.dumps(data)
        response = requests.request("POST", url, data=data, headers=headers)
        return json.loads(json.dumps(response.text))

    def generate_network(self, boxX=100, boxY=100, boxZ=100, proxF=1, \
                         cylChckBox=True, cylD=2, cylDD=1, cylL=30, cylLD=1, cylN=100, \
                         sphChckBox=True, sphD=10, sphDD=1, sphN=10, CFM=0, withAir=False, tag="", threeD=True,
                         steps=0):
        data = {
            "addCylinders": cylChckBox,
            "addSpheres": sphChckBox,
            "boxDimensionX": boxX,
            "boxDimensionY": boxY,
            "boxDimensionZ": boxZ,
            "cfm": CFM,
            "cylindersSettings": {
                "diamDev": cylDD,
                "diameter": cylD,
                "length": cylL,
                "lengthDev": cylLD,
                "number": cylN,
                "sticky": False,
                "enabled": cylChckBox,
            },
            "drawMode": "LINE",
            "dumbbellCylinders": False,
            "highlightContactedObjects": False,
            "polygonsNumber": 5,
            "proximity": proxF,
            "proximitySphere": True,
            "randomCylinderLength": 0,
            "randomCylinderRadius": 0,
            "randomSphereRadius": 0,
            "showAABB": False,
            "showAxes": False,
            "showBox": False,
            "showContacts": False,
            "showObjects": False,
            "spheresSettings": {
                "diamDev": sphDD,
                "diameter": sphD,
                "number": sphN,
                "sticky": False,
                "enabled": sphChckBox
            },
            "steps": steps,
            "tag": tag,
            "withAir": withAir,
            "threeD": threeD
        }
        #         data={
        #           "aabbCheckBoxSelected": True,
        #           "axesCheckBoxSelected": True,
        #           "boxCheckBoxSelected": True,
        #           "boxDimensionXField": boxX,
        #           "boxDimensionYField": boxY,
        #           "boxDimensionZField": boxZ,
        #           "cfm": CFM,
        #           "contactsCheckBoxSelected": True,
        #           "cylindersCheckBoxSelected": cylChckBox,
        #           "cylindersDiamDevField": cylDD,
        #           "cylindersDiameterField": cylD,
        #           "cylindersLengthDevField": cylLD,
        #           "cylindersLengthField": cylL,
        #           "cylindersNumberField": cylN,
        #           "drawModeComboBox": "LINE",
        #           "highlightContactedObjectCheckBoxSelected": True,
        #           "objectsCheckBoxSelected": True,
        #           "polygonsNumberField": 0,
        #           "proximityField": proxF,
        #           "proximitySphereCheckBoxSelected": True,
        #           "randomCylinderDiameter": 0,
        #           "randomCylinderLength": 0,
        #           "randomSphereDiameter": 0,
        #           "spheresCheckBoxSelected": sphChckBox,
        #           "spheresDiamDevField": sphDD,
        #           "spheresDiameterField": sphD,
        #           "spheresNumberField": sphN,
        #           "withAir": withAir,
        #         }
        response = Percolator.generate_network_native(self, data)
        return response

    def analyze(self, withAir=False):
        url = Percolator.serverUrl + 'analyze'
        payload = {'withAir': str(withAir).lower()}
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        response = requests.request("POST", url, headers=headers, params=payload)
        return json.loads(json.dumps(response.text))

    def clear(self):
        url = Percolator.serverUrl + 'clear'
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        response = requests.request("POST", url, headers=headers)
        return json.loads(json.dumps(response.text))

    def export_scene(self):
        url = Percolator.serverUrl + 'export'
        headers = {
            'Accept': 'text/plain',
        }
        response = requests.get(url, headers=headers)
        return json.loads(response.text)

    def export_network(self):
        url = Percolator.serverUrl + 'export-network'
        headers = {
            'Accept': 'text/plain',
        }
        response = requests.get(url, headers=headers)
        return json.loads(response.text)


def generate_net(utils, clear=True, boxX=200, boxY=200, boxZ=200, cylD=2, cylDD=0, cylL=100, cylLD=0, cylN=600,
                 proxF=0.5, cylChckBox=True, sphD=10, sphDD=1, sphN=10, sphChckBox=False, tag="", withAir=False,
                 threeD=True, steps=0):
    if clear:
        utils.clear()
    utils.generate_network(proxF=proxF, boxX=boxX, boxY=boxY, boxZ=boxZ, cylD=cylD, cylDD=cylDD, cylL=cylL, cylLD=cylLD,
                           cylN=cylN, cylChckBox=cylChckBox, sphD=sphD, sphDD=sphDD, sphN=sphN, sphChckBox=sphChckBox,
                           withAir=withAir, threeD=threeD, tag=tag, steps=steps)
    utils.analyze(withAir=withAir)
    network = utils.export_network()
    network['stat']
    network['stat']['aspect'] = cylL / cylD
    network['stat']['boxVol'] = boxX * boxY * boxZ
    network['stat']['cylN'] = cylN
    network['stat']
    return network

def network_create(attempts=5):
    utils = Percolator(serverUrl="http://localhost:8096/percolator/");

    net_len = 0

    box, boy, boz = 1000, 1000, 25

    network = generate_net(utils, clear=True, boxX=box, boxY=boy, boxZ=boz, cylD=0.2, cylL=70, cylN=500, proxF=0.01,
                           threeD=False, sphChckBox=False, cylChckBox=True, tag='Wo3', steps=0)
    # network = generate_net(resutils, clear=True, boxX=box, boxY=boy, boxZ=boz, proxF=0.01,
    #                        threeD=False, sphChckBox=True, cylChckBox=False, sphD=25, sphDD=0, sphN=1500, tag='Wo3')
    network = generate_net(utils, clear=False, boxX=box, boxY=boy, boxZ=boz, cylD=0.2, cylL=400, cylLD=0.2, cylN=150,
                           proxF=0.01, threeD=False, sphChckBox=False, cylChckBox=True, tag='Ag', steps=0)

    G = load_graph_from_json(network)

    els1 = get_electrodes_rects([2, 1], gap=0.3)
    els2 = get_electrodes_rects([3, 1], gap=0.3)
    xmin, xmax, ymin, ymax, zmin, zmax = get_3d_minmax(G)
    delta = 100
    accepted_graphs = get_connected_graphs_electrodes(supergraph=G,delta=delta,elects1=els1,elects2=els2,box=box,boy=boy,boz=boz)
    # accepted_graphs=get_connected_graphs(G,25)
    # accepted_graphs




    el1_nodes = get_nodes_for_electrode(elects=els1, pos=get_pos_for_subgraph(accepted_graphs[0],
                                                                              nx.get_node_attributes(G, 'pos3d')),
                                        xmax=xmin + delta, xmin=xmin - delta, ymax=ymax, zmax=zmax)
    el2_nodes = get_nodes_for_electrode(elects=els2, pos=get_pos_for_subgraph(accepted_graphs[0],
                                                                              nx.get_node_attributes(G, 'pos3d')),
                                        xmax=xmax + delta, xmin=xmax - delta, ymax=ymax, zmax=zmax)

    print(el1_nodes)
    print(el2_nodes)

    # plot_nxgraph(accepted_graphs[1],nx.get_node_attributes(accepted_graphs[1],'pos'))
    graph = accepted_graphs[0]
    title_str = "Edges: {}, TPVF: {:1.3f}, L/R: {}".format(len(graph.edges), network['stat']['TPVF'],
                                                           network['stat']['aspect'])
    # network['stat']['aspect']=cylL*2/cylD
    # network['stat']['boxVol']=boxX*boxY*boxZ
    # network['stat']['cylN']=cylN
    ax = plot_pos3d(graph, title=title_str)
    xmin, xmax, ymin, ymax, zmin, zmax = get_3d_minmax(graph)
    plot_electrodes(xmax=xmin, ymax=ymax, zmax=zmax, ax=ax, els=els1, xdelta=delta)
    plot_electrodes(xmax=xmax, ymax=ymax, zmax=zmax, ax=ax, els=els2, xdelta=delta)
    # plot_pos3d(nx.get_node_attributes(accepted_graphs[1],'pos3d'))

    # plot_nxgraph(G,nx.get_node_attributes(G,'pos'))

    #GROOM GRAPH

    edge_search_cutoff = 25
    source_node = el1_nodes[0][0]  # source node on the electrode
    dest_node = el2_nodes[0][0]  # destination node on the electrode
    # constrained=convert_devices_to_resistors(accepted_graphs[0],min_length=2,ma)
    wired_electrodes_graph = wire_nodes_on_electrodes(electrodeslist=[el1_nodes, el2_nodes],
                                                      subgraph=accepted_graphs[0])

    el_pan = []
    for el_arr in [el1_nodes, el2_nodes]:
        sub_el = []
        for elk in el_arr.keys():
            sub_el.append(el_arr[elk][0])
        el_pan.append(sub_el)

    comb_graph = wired_electrodes_graph
    comb_graph = prune_dead_edges(wired_electrodes_graph, runs=25)
    # comb_graph = convert_devices_to_resistors(comb_graph,min_length=0.00,max_length=50,in_range_dev='m',out_range_dev='r')
    comb_graph = convert_edgeclass_to_device(comb_graph)

    # print(G)
    return comb_graph, el_pan

def network_groomer(accepted_graphs,els1,els2,el1_nodes,el2_nodes,xmin,xmax,ymin,ymax,zmin,zmax,delta):
    edge_search_cutoff = 25
    source_node = el1_nodes[0][0]  # source node on the electrode
    dest_node = el2_nodes[0][0]  # destination node on the electrode
    # constrained=convert_devices_to_resistors(accepted_graphs[0],min_length=2,ma)
    wired_electrodes_graph = wire_nodes_on_electrodes(electrodeslist=[el1_nodes, el2_nodes],
                                                      subgraph=accepted_graphs[0])

    el_pan = []
    for el_arr in [el1_nodes, el2_nodes]:
        sub_el = []
        for elk in el_arr.keys():
            sub_el.append(el_arr[elk][0])
        el_pan.append(sub_el)

    # el_node_pairs = list(itertools.product(el_pan[0], el_pan[1]))
    # for item in list(itertools.combinations(el_pan[0], 2)):
    #     el_node_pairs.append(item)
    # for item in list(itertools.combinations(el_pan[1], 2)):
    #     el_node_pairs.append(item)
    #
    # s_graphs = []
    # for node_pair in tqdm(el_node_pairs):
    #     s_graphs.append(
    #         get_simple_paths_graph_gtalg(G=wired_electrodes_graph, cutoff=edge_search_cutoff, start=node_pair[0],
    #                                      end=node_pair[1]))
    #
    # comb_graph = nx.Graph()
    # for s_graph in s_graphs:
    #     comb_graph.add_nodes_from(s_graph.nodes(data=True))
    #     comb_graph.add_edges_from(s_graph.edges(data=True))
    #
    # nx.set_edge_attributes(comb_graph, nx.get_edge_attributes(wired_electrodes_graph, 'edgetype'), 'edgetype')
    # nx.set_node_attributes(comb_graph, nx.get_node_attributes(wired_electrodes_graph, 'pos'), 'pos')
    # nx.set_node_attributes(comb_graph, nx.get_node_attributes(wired_electrodes_graph, 'pos3d'), 'pos3d')
    #
    # title_str = "Edges: {}".format(len(comb_graph.edges))
    # ax = plot_pos3d(wired_electrodes_graph, title=title_str)
    # plot_electrodes(xmax=xmin, ymax=ymax, zmax=zmax, ax=ax, els=els1, xdelta=delta)
    # plot_electrodes(xmax=xmax, ymax=ymax, zmax=zmax, ax=ax, els=els2, xdelta=delta)

def get_dataset(type='xor',periods=6,boost=1,var=0.0):

    ttables = {}
    ttables['xor'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
    ttables['or'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
    ttables['and'] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

    data=np.array(ttables[type]*periods)
    X = data[:,:-1]
    y = data[:,-1]
    #X,y = make_gaussian_quantiles(n_features=2, n_classes=2,  n_samples=20)
    X = netfitter.perturb_X(X,boost=boost,var=var)

    return X,y

def main():

    X, y = get_dataset(type='xor',periods=6,boost=10,var=0.5)

    # accepted_graphs,el1_nodes,el2_nodes = network_create()
    comb_graph, el_pan =network_create()
    ax = plot_pos3d(comb_graph, title="test")

    ins = el_pan[0][:2]
    outs = el_pan[1][:3]
    # Weird but you have to run this method twice to correctly apply conversions
    # comb_graph = wired_electrodes_graph
    # comb_graph = prune_dead_edges(wired_electrodes_graph, runs=25)
    # comb_graph = convert_devices_to_resistors(comb_graph,min_length=0.00,max_length=6.0,in_range_dev='m',out_range_dev='r')
    mems = sum([v == 'm' for v in nx.get_edge_attributes(comb_graph, 'edgetype').values()])
    print("Total mems: ", mems)
    circ = transform_network_to_circuit(graph=comb_graph, inels=ins, outels=outs, t_step="5e-6", scale=1E-6)

    # with open('/home/nifrick/Documents/development/jupyter/networktest/shortest_path_depth_analysis/circuit1_depth17_dd.json',
    #         'r') as f:
    #     circ=json.loads(json.load(f))
    #
    # ins=circ['inputids']
    # outs=circ['outputids']

    # nf = netfitter.NetworkFitter()
    # nf.eq_time = 0.01
    # circ = modify_integration_time(circ, set_val='1e-5')
    # nf.circuit = circ
    # resx = nf.network_eval(X, y);
    #
    # nf=netfitter.NetworkFitter()
    # resutils = utilities.Utilities(serverUrl=nf.serverUrl)
    # res = {}
    # for n in tqdm(range(3)):
    #     nf = netfitter.NetworkFitter()
    #     nf.eq_time = 0.0001
    #     circ = modify_integration_time(circ, set_val='1e-5')
    #     nf.circuit = circ
    #     res[n] = nf.run_single_sim_series([1500., -1500.], 0, circ['inputids'], circ['outputids'], circ['circuit'],
    #                                       0.001, resutils, repeat=150)
    #
    # t=np.linspace(0,1,40)
    # # sig = signal.sawtooth(2*np.pi*4*t)
    # sig = np.sin(2 * np.pi * 2 * t)
    #
    # X=np.hstack(((sig.reshape((-1, 1))+1)*0+100, -100+0*sig.reshape((-1, 1)),0*sig.reshape((-1, 1))))
    #
    # # res[0] = nf.run_continuous_sim(X,circ['inputids'],circ['outputids'],circ['circuit'],0.0005,resutils)
    #
    # batch_plot_single_sim(res, title='mem 1.0 cutoff', num_elects=3)

if __name__ == "__main__":
    main()

