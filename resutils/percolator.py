import requests
import json
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
# matplotlib.use('QT5Agg')
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
from posixpath import join as urljoin
from collections import OrderedDict
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.cm as cmx
import scipy
import copy
import plotly.graph_objects as go
import requests
import json
import time

class Percolator:
    serverUrl_uuid = ""
    serverUrl=""

    def __init__(self, serverUrl):
        Percolator.serverUrl = serverUrl
        Percolator.serverUrl_uuid = urljoin(serverUrl, "{uuid}/")
        # Percolator.serverUrl_uuid = serverUrl+"uuid/"

    def get_default_config(self):

        defconfig = {
            "cylinder": {
                "angleDev": 0,
                "angleX": 0,
                "angleZ": 0,
                "diamDev": 0,
                "diameter": 0,
                "enabled": False,
                "length": 0,
                "lengthDev": 0,
                "number": 0,
                "sticky": False
            },
            "dumbbell": {
                "angleDev": 0,
                "angleX": 0,
                "angleZ": 0,
                "diamDev": 0,
                "diameter": 0,
                "enabled": False,
                "length": 0,
                "lengthDev": 0,
                "number": 0,
                "sticky": False
            },
            "simulation": {
                "boxDimensionX": 0,
                "boxDimensionY": 0,
                "boxDimensionZ": 0,
                "boxPositionX": 0,
                "boxPositionY": 0,
                "boxPositionZ": 0,
                # "boxDimension":{
                #     "x":100,
                #     "y":100,
                #     "z":100
                # },
                # "boxPosition": {
                #     "x": 0,
                #     "y": 0,
                #     "z": 0
                # },
                "boxAngleX":0,
                "boxAngleY":0,
                "cfm": 1E-11,
                "erp": 0.3,
                "is3D": False,
                "proximity": 2,
                "seed": 0,
                "steps": 0,
                "tag": "string",
                "withAir": False
            },
            "spaghetti": {
                "angleDev": 0,
                "firstAngleDev": 0,
                "angleX": 0,
                "angleZ": 0,
                "diamDev": 0,
                "diameter": 0,
                "enabled": False,
                "length": 0,
                "lengthDev": 0,
                "number": 0,
                "numberOfSegments": 0,
                "sticky": False
            },
            "sphere": {
                "angleDev": 0,
                "angleX": 0,
                "angleZ": 0,
                "diamDev": 0,
                "diameter": 0,
                "enabled": False,
                "number": 0,
                "sticky": False
            # },
            # "visualization": {
            #     "drawMode": "LINE",
            #     "highlightContactedObjects": False,
            #     "polygonsNumber": 0,
            #     "proximitySphere": False,
            #     "showAABB": False,
            #     "showAxes": False,
            #     "showBox": False,
            #     "showContacts": False,
            #     "showObjects": False
            },
            "group": {
                "angleDev": 0,
                "angleX": 0,
                "angleZ": 0,
                "enabled": False,
                "groupsLatticeEnabled": False,
                "groupsLatticeX": 0,
                "groupsLatticeY": 0,
                "groupsLatticeZ": 0,
                "number": 0,
                "sticky": False
            }
        }

        return defconfig

    def create_elect_boxes(self,elmat=[1, 1], gap=0.3, plane=0,
                           box={'x0': 0, 'y0': 0, 'z0': 0, 'x1': 100, 'y1': 100, 'z1': 100}, delta=(0, 0, 0)):
        elects = self.get_electrodes_rects(elmat, gap=gap)
        elects_boxes = {}
        for k in elects.keys():
            if plane == 0:
                y0, z0, h, w = elects[k][0], elects[k][1], elects[k][2], elects[k][3]
                y1 = y0 + h
                z1 = z0 + w
                y0 = y0 * abs(box['y1'] - box['y0']) * 0.9 + box['y0']
                y1 = y1 * abs(box['y1'] - box['y0']) * 0.9 + box['y0']
                z0 = z0 * abs(box['z1'] - box['z0']) * 0.9 + box['z0']
                z1 = z1 * abs(box['z1'] - box['z0']) * 0.9 + box['z0']
                elects_boxes[k] = {'x0': box['x0'] - delta[0], 'x1': box['x0'] + delta[0], 'y0': y0, 'y1': y1, 'z0': z0,
                                   'z1': z1}
            elif plane == 1:
                y0, z0, h, w = elects[k][0], elects[k][1], elects[k][2], elects[k][3]
                y1 = y0 + h
                z1 = z0 + w
                y0 = y0 * abs(box['y1'] - box['y0']) * 0.9 + box['y0']
                y1 = y1 * abs(box['y1'] - box['y0']) * 0.9 + box['y0']
                z0 = z0 * abs(box['z1'] - box['z0']) * 0.9 + box['z0']
                z1 = z1 * abs(box['z1'] - box['z0']) * 0.9 + box['z0']
                elects_boxes[k] = {'x0': box['x1'] - delta[0], 'x1': box['x1'] + delta[0], 'y0': y0, 'y1': y1, 'z0': z0,
                                   'z1': z1}
            elif plane == 2:
                x0, z0, h, w = elects[k][0], elects[k][1], elects[k][2], elects[k][3]
                x1 = x0 + h
                z1 = z0 + w
                x0 = x0 * abs(box['x1'] - box['x0']) * 0.9 + box['x0']
                x1 = x1 * abs(box['x1'] - box['x0']) * 0.9 + box['x0']
                z0 = z0 * abs(box['z1'] - box['z0']) * 0.9 + box['z0']
                z1 = z1 * abs(box['z1'] - box['z0']) * 0.9 + box['z0']
                x0 += 4 * delta[0]
                x1 += 4 * delta[0]
                elects_boxes[k] = {'x0': x0, 'x1': x1, 'y0': box['y0'] - delta[1], 'y1': box['y0'] + delta[1], 'z0': z0,
                                   'z1': z1}
            elif plane == 3:
                x0, z0, h, w = elects[k][0], elects[k][1], elects[k][2], elects[k][3]
                x1 = x0 + h
                z1 = z0 + w
                x0 = x0 * abs(box['x1'] - box['x0']) * 0.9
                x1 = x1 * abs(box['x1'] - box['x0']) * 0.9
                z0 = z0 * abs(box['z1'] - box['z0']) * 0.9
                z1 = z1 * abs(box['z1'] - box['z0']) * 0.9
                x0 += 4 * delta[0]
                x1 += 4 * delta[0]
                elects_boxes[k] = {'x0': x0, 'x1': x1, 'y0': box['y1'] - delta[1], 'y1': box['y1'] + delta[1], 'z0': z0,
                                   'z1': z1}
        return elects_boxes

    def get_graphs_connecting_electrodearrays(self,supergraph, el_arrays=None):
        pos3d = nx.get_node_attributes(supergraph, 'pos3d')
        subgraphs = list(nx.connected_component_subgraphs(supergraph))
        accepted_graphs = []
        nodes_in_earray = {}
        for sg in subgraphs:
            subpos = self.get_pos_for_subgraph(sg, pos3d)
            isgood = []
            for k, el_array in el_arrays.items():
                n_in_arr = self.get_nodes_for_box_array(subpos, el_array)
                for kk, vv in n_in_arr.items():
                    if len(vv) > 0:
                        isgood.append(True)
                    else:
                        isgood.append(False)
            if False not in isgood:
                accepted_graphs.append(sg)
        return accepted_graphs

    def get_nodes_for_box_array(self,pos, el_array=None):
        elects_bucket = {}
        for k in el_array.keys():
            x1, y1, z1, x2, y2, z2 = el_array[k]['x0'], el_array[k]['y0'], el_array[k]['z0'], el_array[k]['x1'], \
                                     el_array[k]['y1'], el_array[k]['z1']
            elects_bucket[k] = self.get_nodes_within_3dboundary(pos, lowerbx=x1, upperbx=x2, lowerby=y1, upperby=y2,
                                                                lowerbz=z1, upperbz=z2)
        return elects_bucket

    def generate_network_native(self, key,data):
        url = Percolator.serverUrl_uuid + 'generate'
        payload = {'uuid': key}
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        data = json.dumps(data)
        response = requests.request("POST", url, data=data, headers=headers,params=payload)
        return json.loads(json.dumps(response.text))

    def create(self):
        url = urljoin(Percolator.serverUrl, 'create')
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        response = requests.request("POST", url, headers=headers)
        return json.loads(json.dumps(response.text))[1:-1]

    def list_uuids(self):
        url = urljoin(Percolator.serverUrl, 'uuids')
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        response = requests.request("POST", url, headers=headers)
        return json.loads(json.dumps(response.text))

    def delete(self, key):
        url = urljoin(Percolator.serverUrl_uuid, 'delete')
        payload = {'uuid':key}
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        response = requests.request("POST", url, headers=headers, params=payload)
        return json.loads(json.dumps(response.text))

    # def generate_network(self, boxX=100, boxY=100, boxZ=100, proxF=1, \
    #                      cylChckBox=True, cylD=2, cylDD=1, cylL=30, cylLD=1, cylN=100, \
    #                      sphChckBox=True, sphD=10, sphDD=1, sphN=10, CFM=0, withAir=False, tag="", threeD=True,
    #                      steps=0):
    def generate_network(self, key, **data):
        # data = {
        #     "addCylinders": cylChckBox,
        #     "addSpheres": sphChckBox,
        #     "boxDimensionX": boxX,
        #     "boxDimensionY": boxY,
        #     "boxDimensionZ": boxZ,
        #     "cfm": CFM,
        #     "cylindersSettings": {
        #         "diamDev": cylDD,
        #         "diameter": cylD,
        #         "length": cylL,
        #         "lengthDev": cylLD,
        #         "number": cylN,
        #         "sticky": False,
        #         "enabled": cylChckBox,
        #     },
        #     "drawMode": "LINE",
        #     "dumbbellCylinders": False,
        #     "highlightContactedObjects": False,
        #     "polygonsNumber": 5,
        #     "proximity": proxF,
        #     "proximitySphere": True,
        #     "randomCylinderLength": 0,
        #     "randomCylinderRadius": 0,
        #     "randomSphereRadius": 0,
        #     "showAABB": False,
        #     "showAxes": False,
        #     "showBox": False,
        #     "showContacts": False,
        #     "showObjects": False,
        #     "spheresSettings": {
        #         "diamDev": sphDD,
        #         "diameter": sphD,
        #         "number": sphN,
        #         "sticky": False,
        #         "enabled": sphChckBox
        #     },
        #     "steps": steps,
        #     "tag": tag,
        #     "withAir": withAir,
        #     "threeD": threeD
        # }

        response = Percolator.generate_network_native(self, key, data)
        return response

    def analyze(self, key,withAir=False):
        url = urljoin(Percolator.serverUrl_uuid, 'analyze')
        payload = {'withAir': str(withAir).lower(),'uuid':key}
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        response = requests.request("POST", url, headers=headers, params=payload)
        return json.loads(json.dumps(response.text))

    def clear(self,key):
        url = urljoin(Percolator.serverUrl_uuid, 'clear')
        payload = {'uuid': key}
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        response = requests.request("POST", url, headers=headers,params=payload)
        return json.loads(json.dumps(response.text))

    def export_scene(self,key):
        url = urljoin(Percolator.serverUrl_uuid, 'export')
        payload = {'uuid': key}
        headers = {
            'Accept': '*/*',
        }
        response = requests.get(url, headers=headers,params=payload)
        return json.loads(response.text)

    def export_network(self,key):
        url = urljoin(Percolator.serverUrl_uuid, 'export-network')
        payload = {'uuid': key}
        headers = {
            'Accept': '*/*',
        }
        response = requests.get(url, headers=headers,params=payload)
        return json.loads(response.text)

    def import_net(self,key,data):
        url = Percolator.serverUrl_uuid + 'import'
        payload = {'uuid': key}
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        data = json.dumps(data)
        response = requests.request("POST", url, data=data, headers=headers,params=payload)
        return json.loads(json.dumps(response.text))

    def get_settings(self,key):
        url = urljoin(Percolator.serverUrl_uuid, 'settings')
        payload = {'uuid': key}
        # headers = {
        #     'Accept': 'text/plain',
        # }
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
        }
        response = requests.get(url, headers=headers,params=payload)
        return json.loads(json.dumps(response.text))


    # def generate_net(self, key,clear=True, boxX=200, boxY=200, boxZ=200, cylD=2, cylDD=0, cylL=100, cylLD=0, cylN=600,
    #                  proxF=0.5, cylChckBox=True, sphD=10, sphDD=1, sphN=10, sphChckBox=False, tag="", withAir=False,
    #                  threeD=True, steps=0):
    #     if clear:
    #         self.clear()
    #     self.generate_network(proxF=proxF, boxX=boxX, boxY=boxY, boxZ=boxZ, cylD=cylD, cylDD=cylDD, cylL=cylL, cylLD=cylLD,
    #                            cylN=cylN, cylChckBox=cylChckBox, sphD=sphD, sphDD=sphDD, sphN=sphN, sphChckBox=sphChckBox,
    #                            withAir=withAir, threeD=threeD, tag=tag, steps=steps)
    #     self.analyze(key=key,withAir=withAir)
    #     network = self.export_network(key)
    #     network['stat']
    #     network['stat']['aspect'] = cylL / cylD
    #     network['stat']['boxVol'] = boxX * boxY * boxZ
    #     network['stat']['cylN'] = cylN
    #     network['stat']
    #     return network


    def generate_net(self,key, **data):
        # if clear:
        #     self.clear()
        self.generate_network(key=key,**data)
        self.analyze(key=key,withAir=data['simulation']['withAir'])
        network = self.export_network(key)
        network['stat']

        network['stat']['boxVol'] = data['simulation']['boxDimensionX'] * \
                                    data['simulation']['boxDimensionY'] * \
                                    data['simulation']['boxDimensionZ']
        try:
            network['stat']['aspect'] = data['cylinder']['length'] / data['cylinder']['diameter']
            network['stat']['cylN'] = data['cylinder']['number']
        except:
            pass

        try:
            network['stat']['sphereD'] = data['sphere']['diameter']
            network['stat']['sphereN'] = data['sphere']['number']
        except:
            pass

        try:
            network['stat']['aspect'] = data['spaghetti']['length'] / data['spaghetti']['diameter']
            network['stat']['cylN'] = data['spaghetti']['number']
        except:
            pass

        network['stat']
        return network

    def get_dataset(self,type='xor',periods=6,boost=1,var=0.0):

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

    def load_graph_from_json_file(self,
            path='/home/nifrick/Documents/development/jupyter/networktest/spatial_percolation_data/16box_5.json'):
        with open(path) as f:
            data = json.load(f)
        graph = self.load_graph_from_json(data)
        return graph

    def load_graph_from_json(self,data):
        G = nx.Graph()
        for k in data['pos'].keys():
            G.add_node(int(k), pos=(data['pos'][k][0], data['pos'][k][1]),
                       pos3d=(data['pos'][k][0], data['pos'][k][1], data['pos'][k][2]))

        for k in data['list']:
            for n in data['list'][k]:
                G.add_edge(int(k), int(n[1]), edgetype='r', edgeclass=n[0])

        pos = nx.get_node_attributes(G, 'pos')
        pos3d = nx.get_node_attributes(G, 'pos3d')

        return G

    def pairwise(self,iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    def get_simple_paths_graph(self,G, start, end, cutoff=None):
        paths = nx.all_simple_paths(G, start, end, cutoff=cutoff)
        sG = nx.Graph()

        pos = nx.get_node_attributes(G, 'pos')
        pos3d = nx.get_node_attributes(G, 'pos3d')

        for path in paths:
            for pair in self.pairwise(path):
                sG.add_edge(pair[0], pair[1])

        nx.set_node_attributes(sG, pos, 'pos')
        nx.set_node_attributes(sG, pos3d, 'pos3d')
        return sG

    def get_simple_paths_graph_gtalg(self,G, start, end, cutoff=None):
        pos = nx.get_node_attributes(G, 'pos')
        pos3d = nx.get_node_attributes(G, 'pos3d')
        gtG = nxutils.nx2gt(nxG=G)
        paths = gt.all_paths(gtG, gt.find_vertex(gtG, gtG.vertex_properties['id'], start)[0],
                             gt.find_vertex(gtG, gtG.vertex_properties['id'], end)[0], cutoff=cutoff)
        #     paths=gt.all_shortest_paths(gtG,gt.find_vertex(gtG,gtG.vertex_properties['id'],start)[0],gt.find_vertex(gtG,gtG.vertex_properties['id'],end)[0])

        sG = nx.Graph()

        for path in paths:
            for pair in self.pairwise(path):
                #             sG.add_edge(pair[0],pair[1])
                sG.add_edge(int(gtG.vertex_properties['id'][pair[0]]), int(gtG.vertex_properties['id'][pair[1]]))

        nx.set_node_attributes(sG, values=pos, name='pos')
        nx.set_node_attributes(sG, values=pos3d, name='pos3d')
        return sG

    def wire_nodes_on_electrodes(self,subgraph, electrodeslist=[]):
        subgraph = subgraph.copy()
        for electrodearray in electrodeslist:
            for electrode in electrodearray.values():
                #             for node in electrodearray[electrode]:
                for wire in self.pairwise(electrode):  # itertools.combinations(electrode,2):
                    subgraph.add_edge(wire[0], wire[1], edgetype='w',edgeclass='wire')
        return subgraph



    def convert_devices_to_resistors(self,graph, min_length=10, max_length=1e8, out_range_dev='m', in_range_dev='r'):
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

    def convert_edgeclass_to_device(self,graph, mem='wo3', res='ag', diode='none'):
        graph_copy = graph.copy()
        edges = graph_copy.edges()
        for e, v in nx.get_edge_attributes(graph_copy, 'edgeclass').items():
            if mem.lower() in v.lower():
                graph_copy[e[0]][e[1]]['edgetype'] = 'm'
            elif res.lower() in v.lower():
                graph_copy[e[0]][e[1]]['edgetype'] = 'r'
            elif diode.lower() in v.lower():
                graph_copy[e[0]][e[1]]['edgetype'] = 'd'
        return graph_copy

    def prune_dead_edges(self,graph, runs=25):
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

    def prune_dead_edges_el_pan(self, graph, el_pan, runs=25):
        G = graph.copy()
        nbs = []
        elect_nodes = list(np.ravel(el_pan))
        for run in range(runs):
            for n in sorted([z for z in sorted(G.degree, key=lambda x: x[1], reverse=False) if z[1] == 1]):
                nbs.append([n[0], list(G.neighbors(n[0]))[0]])

            for nb in nbs:
                if nb[0] not in elect_nodes and nb[1] not in elect_nodes:
                    try:
                        G.remove_edge(nb[0], nb[1])
                    except:
                        pass
            # remove isolates
            G.remove_nodes_from(list(nx.isolates(G)))
        return G

    def get_current_distrib(self,n, cmax=10):
        lst = []
        lst.append(float(-cmax))
        for k in range(n - 1):
            lst.append(cmax / (n - 1))
        return lst

    def precondition_trim_lu(self, g, terminals, cutoff=1e-2):

        terminals = list(np.ravel(terminals))
        lap = nx.laplacian_matrix(g)

        lap = lap + scipy.sparse.csr_matrix(np.eye(lap.shape[0]) * 1e-8)
        lap = scipy.sparse.csc_matrix(lap)

        nodelist = list(g.nodes())

        i = np.zeros(lap.shape[0])
        maxv = np.sqrt(lap.shape[0])
        cur_dist = self.get_current_distrib(len(terminals), maxv)

        for term, current in zip(terminals, cur_dist):
            i[np.argwhere(np.array(nodelist) == term)[0][0]] = current

        splu = scipy.sparse.linalg.splu(lap)
        x = splu.solve(i)

        vattr = {}

        for n, val in enumerate(x):
            vattr[nodelist[n]] = val

        nx.set_node_attributes(g, vattr, 'volt')
        volt_attr = nx.get_node_attributes(g, 'volt')
        g2 = copy.deepcopy(g)

        nrem = 0
        removal_list = []
        for e in g.edges():
            try:
                vdiff = abs(volt_attr[e[0]] - volt_attr[e[1]])
                #         print(vdiff)
                if vdiff < cutoff:
                    nrem += 1
                    g2.remove_edge(*e)
            except:
                print("Trim lu: can't find nodepairs:", e)
                pass
        g2.remove_nodes_from(list(nx.isolates(g2)))
        #         print("edges removed",nrem)
        return g2

    def plot_nxgraph(self,G, pos=None, edge_colors=None):
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

    def plot_nxdegree_hist(self,G, n=10):
        plt.figure(figsize=(5, 5))
        plt.hist(list(dict(nx.degree(G)).values()), n, density=True)
        plt.title("Degree Histogram")
        plt.ylabel("Frequency")
        plt.xlabel("Degree")
        plt.show()

    def plot_nxdegree_log(self,G, ax=None):
        degree_sequence = sorted(dict(nx.degree(G)).values(), reverse=True)  # degree sequence
        # print "Degree sequence", degree_sequence
        dmax = max(degree_sequence)
        plt.figure()
        plt.loglog(degree_sequence, 'b-', marker='o')
        plt.title("Degree rank plot")
        plt.ylabel("degree")
        plt.xlabel("rank")
        plt.show()

    def plot_device_length_histogram(self,graph, attribute='', attribute_value=''):
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

    def get_nodes_within_xboundary(self,pos, lowerb, upperb):
        nodes = []
        if lowerb > upperb:
            lowerb, upperb = upperb, lowerb

        for k, v in zip(pos.keys(), pos.values()):
            if v[0] >= lowerb and v[0] <= upperb:
                nodes.append(k)
        return nodes

    def get_nodes_within_3dboundary(self,pos, lowerbx, upperbx, lowerby, upperby, lowerbz, upperbz):
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

    def get_nodes_for_electrode(self,pos, elects, xmin, xmax, ymax, zmax):
        elects_bucket = {}
        for k in elects.keys():
            y1, z1, h, w = elects[k][0], elects[k][1], elects[k][2], elects[k][3]
            y2 = y1 + h
            z2 = z1 + w
            y1 = y1 * ymax
            y2 = y2 * ymax
            z1 = z1 * zmax
            z2 = z2 * zmax
            elects_bucket[k] = self.get_nodes_within_3dboundary(pos, lowerbx=xmin, upperbx=xmax, lowerby=y1, upperby=y2,
                                                           lowerbz=z1, upperbz=z2)
        return elects_bucket

    def get_pos_for_subgraph(self,subgraph, superpos):
        subpos = {}
        for e in subgraph.nodes():
            if e in superpos.keys():
                subpos[e] = superpos[e]
        return subpos

    # get_pos_for_subgraph(subgraphs[7],pos)

    def get_connected_graphs(self,supergraph, xdelta):
        pos = nx.get_node_attributes(supergraph, 'pos')
        subgraphs = list(nx.connected_component_subgraphs(supergraph))
        xmin = min([k[0] for k in pos.values()])
        xmax = max([k[0] for k in pos.values()])
        accepted_graphs = []
        for sg in subgraphs:
            subpos = self.get_pos_for_subgraph(sg, pos)
            nodesright = self.get_nodes_within_xboundary(subpos, xmax - xdelta, 1e8)
            nodesleft = self.get_nodes_within_xboundary(subpos, -1e8, xmin + xdelta)
            if len(nodesright) > 0 and len(nodesleft) > 0:
                accepted_graphs.append(sg)
        return accepted_graphs

    def get_3d_minmax(self,supergraph,is3d=True):
        pos3d = nx.get_node_attributes(supergraph, 'pos3d')
        xmin = min([k[0] for k in pos3d.values()])
        xmax = max([k[0] for k in pos3d.values()])
        ymin = min([k[1] for k in pos3d.values()])
        ymax = max([k[1] for k in pos3d.values()])
        zmin = min([k[2] for k in pos3d.values()])
        zmax = max([k[2] for k in pos3d.values()]) if is3d else 0.
        return xmin, xmax, ymin, ymax, zmin, zmax

    # retrieves graphs connected to elects1 and elects2 - electrode arrays along x axis
    def get_connected_graphs_electrodes(self,supergraph, elects1, elects2, delta, box, boy, boz):
        pos3d = nx.get_node_attributes(supergraph, 'pos3d')
        subgraphs = list(nx.connected_component_subgraphs(supergraph))
        #     xmin,xmax,ymin,ymax,zmin,zmax = get_3d_minmax(supergraph)
        accepted_graphs = []
        for sg in subgraphs:
            subpos = self.get_pos_for_subgraph(sg, pos3d)

            elects1_bucket = self.get_nodes_for_electrode(subpos, elects1, 0 - delta, 0 + delta, boy, boz)
            elects2_bucket = self.get_nodes_for_electrode(subpos, elects2, box - delta, boy + delta, boy, boz)
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

    def plot_3d_scatter(self,xs, ys, zs, xslabel='x', yslabel='y', zslabel='z', ax=None, colorsMap='jet', title=''):
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

    def plot_pos3d_lightning(self,graph=None, ax=None, title='', is3d=True, plot_wires=True, save_as=None, elev=20, azim=90,
                             max_current=1, cmap='jet', dist=5, max_line_width=5, min_line_width=0.4, electrodes=[]):
        pos3d = nx.get_node_attributes(graph, 'pos3d')
        #     max_current=np.max(np.abs(list(nx.get_edge_attributes(graph,'current').values())))
        #     max_line_width = 5
        #     min_line_width = 0.4
        jet = cm = plt.get_cmap(cmap)
        cNorm = Normalize(vmin=-max_current, vmax=max_current)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

        if ax == None:
            fig = plt.figure(figsize=(10, 10))
            #         ax = fig.gca(projection="3d")
            ax = fig.add_subplot(111, projection='3d')
        #         ax.scatter(xs, ys, zs, c='r', s=5)
        # ax.plot(xs,ys,zs, color='r')
        for e in graph.edges():
            x1 = pos3d[e[0]][0]
            y1 = pos3d[e[0]][1]
            z1 = pos3d[e[0]][2] if is3d else 0.
            x2 = pos3d[e[1]][0]
            y2 = pos3d[e[1]][1]
            z2 = pos3d[e[1]][2] if is3d else 0.
            x = [x1, x2]
            y = [y1, y2]
            z = [z1, z2]
            edgetype = {}

            try:
                edgetype = graph[e[0]][e[1]]['edgetype']
                edgecurrent = graph[e[0]][e[1]]['current']
                #             colorVal = scalarMap.to_rgba(abs(edgecurrent))

                do_plot = True

                if ('w' in edgetype):
                    if not plot_wires:
                        do_plot = False
                        pass

                if do_plot:
                    lw = abs(edgecurrent / max_current) * max_line_width
                    lw = min_line_width if lw < min_line_width else lw

                    if 'm' in edgetype:
                        p = ax.plot(x, y, z, color='r', linewidth=lw)
                    #                     p = ax.scatter(x1, y1, z1, c='r',s=lw)
                    else:
                        p = ax.plot(x, y, z, color='g', linewidth=lw)

            #                 if 'm' in edgetype:
            #                     p = ax.plot(x, y, z, color=scalarMap.to_rgba(abs(edgecurrent)), linewidth=lw)
            #                 else:
            #                     p = ax.plot(x, y, z, color=scalarMap.to_rgba(abs(0)), linewidth=lw)

            #             if 'm' in edgetype:
            #                 ax.plot(x, y, z, c='b', label='memristor')
            #             elif 'r' in edgetype:
            #                 ax.plot(x, y, z, c='m', label='resistor')
            #             elif 'w' in edgetype:
            #                 if plot_wires:
            #                     ax.plot(x, y, z, c='g', label='wire')
            #             elif 'd' in edgetype:
            #                 ax.plot(x, y, z, c='orange', label='diode')
            except:
                ax.plot(x, y, z, c='k')
                pass

        # Remove background axis color
        ax.set_facecolor((0, 0, 0, 0))
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Bonus: To get rid of the grid as well:
        ax.grid(False)

        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.dist = dist
        #     sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=-0.001, vmax=0.001))
        #     plt.colorbar(sm)
        for el, col in zip(electrodes, ['y', 'b', 'g', 'r', 'c', 'm'] * 50):
            self.plot_electrode_boxes(ax=ax, el_array=el, cols=[col])

        if save_as == None:
            plt.show()
        else:
            fig.savefig(save_as)
            plt.close()
        return ax

    def plot_pos3d(self,graph=None, ax=None, title='', is3d=True,plot_wires=True):
        pos3d = nx.get_node_attributes(graph, 'pos3d')
        # xs = [pos3d[k][0] for k in accepted_graph.nodes()]
        # ys = [pos3d[k][1] for k in accepted_graph.nodes()]
        # zs = [pos3d[k][2] for k in accepted_graph.nodes()]

        if ax == None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca(projection="3d")
        #         ax.scatter(xs, ys, zs, c='r', s=5)
        # ax.plot(xs,ys,zs, color='r')
        for e in graph.edges():
            x1 = pos3d[e[0]][0]
            y1 = pos3d[e[0]][1]
            z1 = pos3d[e[0]][2] if is3d else 0.
            x2 = pos3d[e[1]][0]
            y2 = pos3d[e[1]][1]
            z2 = pos3d[e[1]][2] if is3d else 0.
            x = [x1, x2]
            y = [y1, y2]
            z = [z1, z2]
            edgetype = {}
            try:
                edgetype = graph[e[0]][e[1]]['edgetype']
                if 'm' in edgetype:
                    ax.plot(x, y, z, c='b', label='memristor')
                elif 'r' in edgetype:
                    ax.plot(x, y, z, c='m', label='resistor')
                elif 'w' in edgetype:
                    if plot_wires:
                        ax.plot(x, y, z, c='g', label='wire')
                elif 'd' in edgetype:
                    ax.plot(x, y, z, c='orange', label='diode')
            except:
                ax.plot(x, y, z, c='k')
                pass

        # Remove background axis color
        ax.set_facecolor((0, 0, 0, 0))
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Bonus: To get rid of the grid as well:
        ax.grid(False)

        plt.title(title)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=20, azim=90)

        plt.show()
        return ax

    def plotly_pos3d(self,graph=None, fig=None, title='', is3d=True, plot_wires=True, memcolor='red', rescolor='goldenrod',
                     wirecolor='green', diodecolor='lightsteelblue', othercolor='teal'):
        pos3d = nx.get_node_attributes(graph, 'pos3d')
        mxl = []
        myl = []
        mzl = []
        mlb = []
        mcl = []

        rxl = []
        ryl = []
        rzl = []
        rlb = []
        rcl = []

        wxl = []
        wyl = []
        wzl = []
        wlb = []
        wcl = []

        dxl = []
        dyl = []
        dzl = []
        dlb = []
        dcl = []

        kxl = []
        kyl = []
        kzl = []
        klb = []
        kcl = []
        if fig == None:
            fig = go.Figure()

        for e in graph.edges():
            x1 = pos3d[e[0]][0]
            y1 = pos3d[e[0]][1]
            z1 = pos3d[e[0]][2] if is3d else 0.
            x2 = pos3d[e[1]][0]
            y2 = pos3d[e[1]][1]
            z2 = pos3d[e[1]][2] if is3d else 0.
            x = [x1, x2, None]
            y = [y1, y2, None]
            z = [z1, z2, None]
            #         xl=xl+x
            #         yl=yl+y
            #         zl=zl+z
            edgetype = {}
            try:
                edgetype = graph[e[0]][e[1]]['edgetype']
                if 'm' in edgetype:
                    mxl = mxl + x
                    myl = myl + y
                    mzl = mzl + z
                    mcl = mcl + [1, 1, 0]
                    mlb = mlb + ['m', 'm', 0]
                elif 'r' in edgetype:
                    rxl = rxl + x
                    ryl = ryl + y
                    rzl = rzl + z
                    rcl = rcl + [2, 2, 0]
                    rlb = rlb + ['r', 'r', 0]
                elif 'w' in edgetype:
                    if plot_wires:
                        wxl = wxl + x
                        wyl = wyl + y
                        wzl = wzl + z
                        wcl = wcl + [3, 3, 0]
                        wlb = wlb + ['w', 'w', 0]
                elif 'd' in edgetype:
                    dxl = dxl + x
                    dyl = dyl + y
                    dzl = dzl + z
                    dcl = dcl + [4, 4, 0]
                    dlb = dlb + ['d', 'd', 0]
            except:
                kxl = kxl + x
                kyl = kyl + y
                kzl = kzl + z
                cl = cl + [5, 5, 0]
                lb = lb + ['k', 'k', 0]
                pass

        fig.add_trace(go.Scatter3d(
            x=mxl, y=myl, z=mzl,
            name='Memristors',
            marker=dict(
                size=0.1,
            ),
            line=dict(
                color=memcolor,
                #         color=[0 if v is None else -v for v in mcl],
                #         colorscale='Viridis',
                width=2
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=rxl, y=ryl, z=rzl,
            name='Resistors',
            marker=dict(
                size=0.1,
            ),
            line=dict(
                color=rescolor,
                #         color=[0 if v is None else -v for v in rcl],
                #         colorscale='Viridis',
                width=2
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=wxl, y=wyl, z=wzl,
            name='Wires',
            marker=dict(
                size=0.1,
            ),
            line=dict(
                color=wirecolor,
                #         color=[0 if v is None else -v for v in wcl],
                #         colorscale='Viridis',
                width=2
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=dxl, y=dyl, z=dzl,
            name='Diodes',
            marker=dict(
                size=0.1,
            ),
            line=dict(
                color=diodecolor,
                #         color=[0 if v is None else -v for v in dcl],
                #         colorscale='Viridis',
                width=2
            )
        ))

        fig.add_trace(go.Scatter3d(
            x=kxl, y=kyl, z=kzl,
            name='Other',
            marker=dict(
                size=0.1,
            ),
            line=dict(
                color=othercolor,
                #         color=[0 if v is None else -v for v in kcl],
                #         colorscale='Viridis',
                width=2
            )
        ))

        fig.update_layout(
            width=800,
            height=700,
            autosize=False,
            scene=dict(
                camera=dict(
                    up=dict(
                        x=0,
                        y=0,
                        z=1
                    ),
                    eye=dict(
                        x=0,
                        y=1.0707,
                        z=1,
                    )
                ),
                aspectratio=dict(x=1, y=1, z=0.7),
                aspectmode='manual',
                bgcolor='rgba(0,0,0,0)'
            ),
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        return fig

    def plot_electrodes(self,ax=None, els=None, xmax=1, ymax=1, zmax=1, xdelta=1):
        x1, x2 = xmax - xdelta, xmax + xdelta
        colors = ['k', 'g', 'b', 'r', 'c', 'm', 'y', 'k'] * 10
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

    def plot_electrode_boxes(self,ax=None, el_array=None, cols=['k', 'g', 'b', 'r', 'c', 'm', 'y', 'k']):
        #     x1, x2 = xmax - xdelta, xmax + xdelta
        colors = cols * 10
        if ax == None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca(projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        for k, n in zip(list(el_array.keys()), range(len(list(el_array.keys())))):
            x1, y1, z1, x2, y2, z2 = el_array[k]['x0'], el_array[k]['y0'], el_array[k]['z0'], el_array[k]['x1'], \
                                     el_array[k]['y1'], el_array[k]['z1']
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

            for e in edges:
                y = np.array(e)[:, 0].tolist()
                z = np.array(e)[:, 1].tolist()
                x = np.array(e)[:, 2].tolist()
                ax.plot(x, y, z, colors[n])
        return ax

    def plotly_electrode_boxes(self,fig=None, el_array=None, cols=[10]):
        #     x1, x2 = xmax - xdelta, xmax + xdelta
        colors = cols * 10
        if fig == None:
            fig = go.Figure()

        for k, n in zip(list(el_array.keys()), range(len(list(el_array.keys())))):
            x1, y1, z1, x2, y2, z2 = el_array[k]['x0'], el_array[k]['y0'], el_array[k]['z0'], el_array[k]['x1'], \
                                     el_array[k]['y1'], el_array[k]['z1']
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

            xl = []
            yl = []
            zl = []
            for e in edges:
                y = np.array(e)[:, 0].tolist()
                z = np.array(e)[:, 1].tolist()
                x = np.array(e)[:, 2].tolist()
                xl = xl + [x[0], x[1], None]
                yl = yl + [y[0], y[1], None]
                zl = zl + [z[0], z[1], None]
            #             ax.plot(x, y, z, colors[n])
            fig.add_trace(
                go.Scatter3d(
                    x=xl, y=yl, z=zl,
                    name="Electrode",
                    marker=dict(
                        size=0.1,
                    ),
                    line=dict(
                        color=colors[n],
                        colorscale='Viridis',
                        width=2
                    )
                ))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        #     fig.show()
        return fig

    def get_electrodes_rects(self,els=[1, 1], gap=0.2):
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



def network_create(attempts=5):
    percolator = Percolator(serverUrl="http://152.14.71.96:8096/percolator/");

    net_len = 0

    box, boy, boz = 1000, 1000, 25

    network = percolator.generate_net(clear=True, boxX=box, boxY=boy, boxZ=boz, cylD=0.2, cylL=70, cylN=300, proxF=0.01,
                           threeD=False, sphChckBox=False, cylChckBox=True, tag='Wo3', steps=0)
    # network = generate_net(resutils, clear=True, boxX=box, boxY=boy, boxZ=boz, proxF=0.01,
    #                        threeD=False, sphChckBox=True, cylChckBox=False, sphD=25, sphDD=0, sphN=1500, tag='Wo3')
    network = percolator.generate_net(clear=False, boxX=box, boxY=boy, boxZ=boz, cylD=0.2, cylL=400, cylLD=0.2, cylN=150,
                           proxF=0.01, threeD=False, sphChckBox=False, cylChckBox=True, tag='Ag', steps=0)

    G = percolator.load_graph_from_json(network)

    els1 = percolator.get_electrodes_rects([2, 1], gap=0.3)
    els2 = percolator.get_electrodes_rects([3, 1], gap=0.3)
    xmin, xmax, ymin, ymax, zmin, zmax = percolator.get_3d_minmax(G)
    delta = 100
    accepted_graphs = percolator.get_connected_graphs_electrodes(supergraph=G,delta=delta,elects1=els1,elects2=els2,box=box,boy=boy,boz=boz)
    # accepted_graphs=get_connected_graphs(G,25)
    # accepted_graphs




    el1_nodes = percolator.get_nodes_for_electrode(elects=els1, pos=percolator.get_pos_for_subgraph(accepted_graphs[0],
                                                                              nx.get_node_attributes(G, 'pos3d')),
                                        xmax=xmin + delta, xmin=xmin - delta, ymax=ymax, zmax=zmax)
    el2_nodes = percolator.get_nodes_for_electrode(elects=els2, pos=percolator.get_pos_for_subgraph(accepted_graphs[0],
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
    ax = percolator.plot_pos3d(graph, title=title_str)
    xmin, xmax, ymin, ymax, zmin, zmax = percolator.get_3d_minmax(graph)
    percolator.plot_electrodes(xmax=xmin, ymax=ymax, zmax=zmax, ax=ax, els=els1, xdelta=delta)
    percolator.plot_electrodes(xmax=xmax, ymax=ymax, zmax=zmax, ax=ax, els=els2, xdelta=delta)
    # plot_pos3d(nx.get_node_attributes(accepted_graphs[1],'pos3d'))

    # plot_nxgraph(G,nx.get_node_attributes(G,'pos'))

    #GROOM GRAPH

    edge_search_cutoff = 25
    source_node = el1_nodes[0][0]  # source node on the electrode
    dest_node = el2_nodes[0][0]  # destination node on the electrode
    # constrained=convert_devices_to_resistors(accepted_graphs[0],min_length=2,ma)
    wired_electrodes_graph = percolator.wire_nodes_on_electrodes(electrodeslist=[el1_nodes, el2_nodes],
                                                      subgraph=accepted_graphs[0])

    el_pan = []
    for el_arr in [el1_nodes, el2_nodes]:
        sub_el = []
        for elk in el_arr.keys():
            sub_el.append(el_arr[elk][0])
        el_pan.append(sub_el)

    comb_graph = wired_electrodes_graph
    comb_graph = percolator.prune_dead_edges(wired_electrodes_graph, runs=25)
    # comb_graph = convert_devices_to_resistors(comb_graph,min_length=0.00,max_length=50,in_range_dev='m',out_range_dev='r')
    comb_graph = percolator.convert_edgeclass_to_device(comb_graph)

    # print(G)
    return comb_graph, el_pan

def network_groomer(self,accepted_graphs,els1,els2,el1_nodes,el2_nodes,xmin,xmax,ymin,ymax,zmin,zmax,delta):
    edge_search_cutoff = 25
    source_node = el1_nodes[0][0]  # source node on the electrode
    dest_node = el2_nodes[0][0]  # destination node on the electrode
    # constrained=convert_devices_to_resistors(accepted_graphs[0],min_length=2,ma)
    wired_electrodes_graph = self.wire_nodes_on_electrodes(electrodeslist=[el1_nodes, el2_nodes],
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

def main():
    # percolator = Percolator(serverUrl="http://spartan.mse.ncsu.edu:8096/percolator/")
    percolator = Percolator(serverUrl="http://spartan.mse.ncsu.edu:15850/percolator/")
    X, y = percolator.get_dataset(type='xor',periods=6,boost=10,var=0.5)

    # # accepted_graphs,el1_nodes,el2_nodes = network_create()
    # comb_graph, el_pan = network_create()
    # ax = percolator.plot_pos3d(comb_graph, title="test")
    #
    # ins = el_pan[0][:2]
    # outs = el_pan[1][:3]
    # # Weird but you have to run this method twice to correctly apply conversions
    # # comb_graph = wired_electrodes_graph
    # # comb_graph = prune_dead_edges(wired_electrodes_graph, runs=25)
    # # comb_graph = convert_devices_to_resistors(comb_graph,min_length=0.00,max_length=6.0,in_range_dev='m',out_range_dev='r')
    # mems = sum([v == 'm' for v in nx.get_edge_attributes(comb_graph, 'edgetype').values()])
    # print("Total mems: ", mems)
    # circ = transform_network_to_circuit(graph=comb_graph, inels=ins, outels=outs, t_step="5e-6", scale=1E-6)

    with open('/home/nifrick/Documents/development/jupyter/networktest/shortest_path_depth_analysis/circuit1_depth17_dd.json',
            'r') as f:
        circ=json.loads(json.load(f))

    ins=circ['inputids']
    outs=circ['outputids']

    # nf = netfitter.NetworkFitter(serverUrl="http://spartan.mse.ncsu.edu:8090/symphony/")
    nf = netfitter.NetworkFitter(serverUrl="http://landau-nic0.mse.ncsu.edu:15833/symphony/")
    nf.eq_time = 0.01
    circ = modify_integration_time(circ, set_val='1e-5')
    nf.circuit = circ
    resx = nf.network_eval(X, y);
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

