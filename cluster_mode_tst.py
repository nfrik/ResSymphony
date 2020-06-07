# %matplotlib ipympl
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from resutils import graph2json as g2json
from mpl_toolkits.mplot3d import Axes3D
from resutils import percolator
import networkx as nx
from tqdm import tqdm_notebook, tnrange
import time
import pandas as pd
from resutils import netfitter2 as netfitter
from resutils import utilities
from resutils import plott
# from tqdm import tqdm
import requests
import json
import itertools
import scipy
import random
import ray

nf_cpu = netfitter.NetworkFitter(serverUrl="http://landau-nic0.mse.ncsu.edu:15834/symphony/")
nf_lancuda = netfitter.NetworkFitter(serverUrl="http://landau-nic0.mse.ncsu.edu:15832/symphony/")
perc = percolator.Percolator(serverUrl="http://landau-nic0.mse.ncsu.edu:15850/percolator/")

import ray
import jsonpickle
import copy

# server_pool=["http://landau-nic0.mse.ncsu.edu:15847/percolator/",
#              "http://kolmogorov-nic0.mse.ncsu.edu:15846/percolator/"]

# create actors based on server pool
ray.shutdown()
if ray.is_initialized() != True:
    ray.init()


def choice2grid(dictionary):
    import itertools
    #     experiment_grid={'VF':[1,2],'N':[3,4],'primaryAngle':[1,2,3],'secondaryAngle':[4,5],'conductivity':[1]}

    # np.array(list(itertools.product(*[[1,2,3],[3,4,5],[3,3]])))
    grid = {}
    items = []
    key_order = []
    for key in dictionary.keys():
        key_order.append(key)
        grid[key] = []
        items.append(dictionary[key])

    items = np.array(list(itertools.product(*items)))
    for n, key in zip(range(len(key_order)), key_order):
        grid[key] = list(items[:, n])

    df = pd.DataFrame(grid)
    return df


@ray.remote
class NetWorker2(object):
    #     def __init__(self,perc_server_url,nf_lancuda,left_electrode,right_electrode,writer,circ_workers):
    def __init__(self, perc_server_url, nf_lancuda, writer, circ_workers):
        self.srv_url = perc_server_url
        self.nf_lancuda = nf_lancuda
        self.perc = percolator.Percolator(serverUrl=self.srv_url)

        self.intermediate_state = []
        self.writer = writer
        self.circ_workers = circ_workers

    def get_params(self):
        return {'perc_srv_url': self.srv_url}

    def func1(self, parameter):
        #         print('time:{} func1 server:{} start sleep:{} for parameter:{}'.format(time.time()*1000,self.srv_url,self.delay,parameter))
        time.sleep(self.delay)
        return 1

    def create_spaghetti_network(self, key=None, N=200, L=100, angleX=0, angleY=0, nsegments=10, lengthDev=0, D=0.1,
                                 diamDev=0, proximity=0.1, primaryAngleDev=10, angleDev=10, seed=0, box=100, boy=100,
                                 boz=100, air=False, is3D=True, tag='Ag'):

        datac = self.perc.get_default_config()
        datac['spaghetti']['enabled'] = True
        datac['spaghetti']['angleX'] = angleX
        datac['spaghetti']['angleZ'] = angleY
        datac['spaghetti']['angleDev'] = angleDev
        datac['spaghetti']['firstAngleDev'] = primaryAngleDev
        datac['spaghetti']['diamDev'] = diamDev
        datac['spaghetti']['number'] = N
        datac['spaghetti']['diameter'] = D
        datac['spaghetti']['length'] = L
        datac['spaghetti']['numberOfSegments'] = nsegments
        datac['spaghetti']['lengthDev'] = lengthDev

        # For now, stats module runs statistics only for cylinders, not spaghetti, but we still need to define cylinder just for the purpose of bypassing division by zero error
        # Here we just need to copy and paste parameters from spaghetti network and disable cylinders. This will be fixed in future versions.
        datac['cylinder']['enabled'] = False
        datac['cylinder']['angleX'] = 0
        datac['cylinder']['angleZ'] = 0
        datac['cylinder']['angleDev'] = angleDev
        datac['cylinder']['diamDev'] = diamDev
        datac['cylinder']['number'] = N
        datac['cylinder']['diameter'] = D
        datac['cylinder']['length'] = L
        datac['cylinder']['lengthDev'] = lengthDev

        datac['simulation']['boxDimensionX'] = box
        datac['simulation']['boxDimensionY'] = boy
        datac['simulation']['boxDimensionZ'] = boz
        datac['simulation']['proximity'] = proximity
        datac['simulation']['seed'] = seed
        datac['simulation']['withAir'] = air
        datac['simulation']['is3D'] = is3D
        datac['simulation']['steps'] = 0
        datac['simulation']['tag'] = tag
        datac['simulation']['isEllipsoidal'] = False

        #         key1=self.perc.create()

        #         network = self.perc.generate_net(key1,**datac)

        #         self.perc.clear(key1)

        #         self.perc.delete(key1)

        #         return network

        if key != None:
            #         network = perc.generate_net(key,**datac)
            self.perc.generate_network(key, **datac)
        else:
            print("No key provided")
        #     perc.clear(key1)
        # curset = perc.export_network(key1)
        # print(curset)
        #     perc.delete(key1)

        return datac

    def create_cylinder_network(self, key=None, N=200, L=100, lengthDev=0, D=0.1, diamDev=0, proximity=0.1, angleDev=90,
                                seed=0, box=100, boy=100, boz=100, steps=1, is3D=True, tag='string', air=False):

        datac = self.perc.get_default_config()
        datac['cylinder']['enabled'] = True
        datac['cylinder']['angleX'] = 0
        datac['cylinder']['angleZ'] = 0
        datac['cylinder']['angleDev'] = angleDev
        datac['cylinder']['diamDev'] = diamDev
        datac['cylinder']['number'] = N
        datac['cylinder']['diameter'] = D
        datac['cylinder']['length'] = L
        datac['cylinder']['lengthDev'] = lengthDev
        datac['simulation']['boxDimensionX'] = box
        datac['simulation']['boxDimensionY'] = boy
        datac['simulation']['boxDimensionZ'] = boz
        datac['simulation']['proximity'] = proximity
        datac['simulation']['seed'] = seed
        datac['simulation']['withAir'] = air
        datac['simulation']['is3D'] = is3D
        datac['simulation']['tag'] = tag
        datac['simulation']['steps'] = steps
        datac['simulation']['erp'] = 0.5

        #     key1=perc.create()
        #     print(key1)
        if key != None:
            #         network = perc.generate_net(key,**datac)
            self.perc.generate_network(key, **datac)
        else:
            print("No key provided")
        #     perc.clear(key1)
        # curset = perc.export_network(key1)
        # print(curset)
        #     perc.delete(key1)

        return datac

    def create_disk_wire_network(self, key, box=400, boy=400, boz=50, diskN=500, diskL=60, diskD=0.1, wireN=500,
                                 wireL=30, wireD=0.1, angleDev=180, proximity=0.05, steps=1, is3D=False, seed=0,
                                 air=False):

        network = None
        retries = 1
        success = False
        while not success:
            try:
                #                 key=self.create_key()
                data = self.create_cylinder_network(key=key, N=diskN, L=diskL, lengthDev=0, D=diskD, diamDev=0,
                                                    proximity=proximity, angleDev=angleDev, seed=seed, box=box, boy=boy,
                                                    boz=boz, steps=steps, is3D=is3D, air=air, tag='Ag')
                data = self.create_cylinder_network(key=key, N=wireN, L=wireL, lengthDev=0, D=wireD, diamDev=0,
                                                    proximity=proximity, angleDev=angleDev, seed=seed, box=box, boy=boy,
                                                    boz=boz, steps=steps, is3D=is3D, air=air, tag='Ag')
                network = self.analyze(key=key, data=data)
                self.destroy_key(key)
                success = True
            except Exception as e:
                self.destroy_key(key)
                wait = retries * 5;
                print('Error! Waiting {} secs and re-trying...'.format(wait))
                time.sleep(wait)
                retries += 1

        #         key=self.create_key()
        #         data=self.create_cylinder_network(key=key,N=diskN,L=diskL,lengthDev=0,D=diskD,diamDev=0,proximity=proximity,angleDev=angleDev,seed=seed,box=box,boy=boy,boz=boz,steps=steps,is3D=is3D,air=air,tag='disk')
        #         data=self.create_cylinder_network(key=key,N=wireN,L=wireL,lengthDev=0,D=wireD,diamDev=0,proximity=proximity,angleDev=angleDev,seed=seed,box=box,boy=boy,boz=boz,steps=steps,is3D=is3D,air=air,tag='wire')
        #         network=self.analyze(key=key,data=data)
        #         self.destroy_key(key)
        return network

    #     def create_spaghetti_network(self,key=None,N=200,L=100,angleX=0,angleY=0,nsegments=10,lengthDev=0,D=0.1,diamDev=0,proximity=0.1,primaryAngleDev=10,angleDev=10,seed=0,box=100,boy=100,boz=100,air=False,tag='Ag'):
    def create_straight_and_curved_wire_network(self, box=100, boy=100, boz=100, proximity=0.0014, \
                                                cwireN=500, cwireL=60, cwireD=0.1, cangleX=0, cangleY=0, nsegments=10,
                                                clengthDev=0, cDiam=0, cdiamDev=0, cprimaryAngleDev=10, cangleDev=10,
                                                ctag='Ag', \
                                                wireN=500, wireL=30, wireD=0.1, angleDev=180, steps=1, is3D=False,
                                                seed=0, air=False, tag='Ag'):

        network = None
        retries = 1
        success = False
        while not success:
            try:
                key = self.create_key()
                data = self.create_spaghetti_network(key=key, N=cwireN, L=cwireL, angleX=cangleX, angleY=cangleY,
                                                     nsegments=nsegments, lengthDev=clengthDev, D=cDiam,
                                                     diamDev=cdiamDev, proximity=proximity,
                                                     primaryAngleDev=cprimaryAngleDev, angleDev=cangleDev, seed=seed,
                                                     box=box, boy=boy, boz=boz, air=air, tag=ctag)
                data = self.create_cylinder_network(key=key, N=wireN, L=wireL, lengthDev=0, D=wireD, diamDev=0,
                                                    proximity=proximity, angleDev=angleDev, seed=seed, box=box, boy=boy,
                                                    boz=boz, steps=steps, is3D=is3D, air=air, tag=tag)
                network = self.analyze(key=key, data=data)
                self.destroy_key(key)
                success = True
            except Exception as e:
                self.destroy_key(key)
                wait = retries * 5;
                print('Error! Waiting {} secs and re-trying...'.format(wait))
                time.sleep(wait)
                retries += 1
        return network

    def create_key(self):
        return self.perc.create()

    def destroy_key(self, key):
        if key != None:
            self.perc.clear(key)
            self.perc.delete(key)

    def analyze(self, key, data):
        self.perc.analyze(key=key, withAir=data['simulation']['withAir'])
        network = self.perc.export_network(key)
        return network

    def is_conducting(self, network, left_electrode, right_electrode):
        G = self.perc.load_graph_from_json(network)
        accepted_graphs = self.perc.get_graphs_connecting_electrodearrays(G, {0: left_electrode, 1: right_electrode})

        if len(accepted_graphs) > 0:
            return accepted_graphs[0]
        else:
            return 0.

    def get_number_edges(self, network):
        G = self.perc.load_graph_from_json(network)
        return len(G.edges())

    def wire_electrodes(self, accepted_graph, mem='Zn', res='Ag', left_electrode=None, right_electrode=None):
        input_nodes = self.perc.get_nodes_for_box_array(pos=nx.get_node_attributes(accepted_graph, 'pos3d'),
                                                        el_array=left_electrode)
        output_nodes = self.perc.get_nodes_for_box_array(pos=nx.get_node_attributes(accepted_graph, 'pos3d'),
                                                         el_array=right_electrode)

        wired_electrodes_graph = self.perc.wire_nodes_on_electrodes(accepted_graph, [input_nodes, output_nodes])

        el_pan = []
        for el_arr in [input_nodes, output_nodes]:
            sub_el = []
            for elk in el_arr.keys():
                sub_el.append(el_arr[elk][0])
            el_pan.append(sub_el)

        comb_graph = wired_electrodes_graph
        comb_graph = self.perc.prune_dead_edges_el_pan(wired_electrodes_graph, el_pan, runs=32)
        comb_graph = self.precondition_trim_lu(comb_graph, el_pan, cutoff=5e-3)
        comb_graph = self.perc.convert_edgeclass_to_device(comb_graph, mem=mem, res=res, diode='sph')

        # self.el_pan = el_pan
        return comb_graph, el_pan

    def convert_2_circuit(self, comb_graph, junction_res=8000,el_pan=None):
        ins = el_pan[0][:]
        outs = el_pan[1][:]
        circ = g2json.transform_network_to_circuit_res_cutoff(graph=comb_graph, inels=ins, outels=outs, Ron_pnm=1,
                                                              Roff_pnm=1000, mobility=2.56e-9, nw_res_per_nm=0.005,
                                                              t_step="5e-5", scale=1e-6, junct_res_per_nm=junction_res,
                                                              mem_cutoff_len_nm=0.1)
        return circ

    def get_conductivity_4_network(self, network, junction_res, box, boy, boz, is3D):
        left_electrode = perc.create_elect_boxes(elmat=[1, 1], plane=0, gap=0.0,
                                                 box={'x0': -5, 'y0': 0, 'z0': 0, 'x1': box, 'y1': boy, 'z1': boz},
                                                 delta=(5, 0, 0))
        right_electrode = perc.create_elect_boxes(elmat=[1, 1], plane=1, gap=0.0,
                                                  box={'x0': 0, 'y0': 0, 'z0': 0, 'x1': box + 5, 'y1': boy, 'z1': boz},
                                                  delta=(5, 0, 0))
        gnet = self.is_conducting(network, left_electrode, right_electrode)
        wired_net = ''
        if type(gnet) != float:
            wired_net, el_pan = self.wire_electrodes(accepted_graph=gnet,left_electrode=left_electrode,right_electrode=right_electrode)
            circuit_net = self.convert_2_circuit(wired_net, junction_res, el_pan)
            #             cond=self.get_conductivity(circuit_net)
            circ_worker = np.random.choice(self.circ_workers)
            cond = ray.get(circ_worker.get_conductivity.remote(circuit_net, box, boy, boz, is3D))
            #             cond=1.
            return cond, 1.
        else:
            return gnet, 0.

    # def get_conductivity_4_network_serial(self, network, junction_res, box, boy, boz, is3D):
    #     gnet = self.is_conducting(network)
    #     wired_net = ''
    #     if type(gnet) != float:
    #         wired_net = self.wire_electrodes(gnet)
    #         circuit_net = self.convert_2_circuit(wired_net, junction_res)
    #         cond = self.get_conductivity(circuit_net, is3D)
    #         return cond, 1.
    #     else:
    #         return gnet, 0.

    def calculate_N4VF_cylinders(self, VF=0.002, L=100, D=0.1, box=1000, boy=1000, boz=1000):
        #     L=int(np.ceil(AR*D))
        #     L=D/AR
        N = int(np.ceil(4.0 * VF * box * boy * boz / math.pi / D ** 2.0 / L))
        return N

    def calculate_LN_surf(self, SF=0.002, L=100, D=0.1, box=1000, boy=1000):
        N = int(round(SF * box * boy / L / D))
        return N

    def get_current_distrib(self, n, cmax=10):
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
                print("Can't find nodepairs:", e)
                pass
        g2.remove_nodes_from(list(nx.isolates(g2)))
        #         print("edges removed",nrem)
        return g2

    def calculate_sparsity(self, A):
        return 1.0 - A.count_nonzero() / float(A.shape[0] * A.shape[1])

    def wire_network_wrapper(self, L, D, VF, junction_res, angle, box, boy, boz,is3D=True):
        retries = 1
        key = None
        N = self.calculate_N4VF_cylinders(VF=VF, L=L, D=D, box=box, boy=boy, boz=boz)
        PVOL = N * np.pi / 4 * D ** 2 * L
        success = False
        while not success:
            try:
                key = self.create_key()
                seed = np.random.randint(0, np.iinfo(np.int32).max)
                #                 data_id=self.create_spaghetti_network(key,N=N,L=20,nsegments=2,lengthDev=0,D=D,diamDev=0,proximity=0.0014,primaryAngleDev=180,angleDev=0,box=box,boy=boy,boz=boz,seed=0,air=False,tag='Ag')
                #                 data_id=self.create_spaghetti_network(key,N=N,L=L,nsegments=5,lengthDev=0,D=D,diamDev=0,proximity=0.0014,primaryAngleDev=180,angleDev=angle,box=box,boy=boy,boz=boz,seed=seed,air=False,tag='Ag')
                data_id = self.create_cylinder_network(key=key, N=N, L=L, lengthDev=0, D=D, diamDev=0, proximity=0.0014,
                                                       angleDev=angle, seed=seed, box=box, boy=boy, boz=boz, steps=0,
                                                       is3D=True, tag='Ag', air=False)
                network_id = self.analyze(key, data_id)

                success = True
            except Exception as e:
                wait = retries * 5;
                print('Error connecting to server:{} ! Waiting {} secs and re-trying...'.format(self.srv_url, wait))
                print("Exception: ", e)
                time.sleep(wait)
                retries += 1
        self.destroy_key(key)
        cond, conducts = self.get_conductivity_4_network(network_id, junction_res, box, boy, boz, is3D)

        #         #check for edge case
        #         if cond==0 and conducts==1:
        #             jsp_network=jsonpickle.encode(network_id)
        jsp_data = jsonpickle.encode(data_id)
        #             with open('edge_network-{}.jsp'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]),'w') as f:
        #                 f.write(jsp_network)
        #             with open('edge_data-{}.jsp'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]),'w') as f:
        #                 f.write(jsp_data)
        edges = self.get_number_edges(network_id)
        #         self.writer.add_data.remote(self.srv_url,VF,180,angle,cond,conducts,0,jsp_data)
        self.writer.add_data.remote(
            {'server': self.srv_url, 'VF': VF, 'PVOL': PVOL, 'N': N, 'primaryAngle': 180, 'conductivity': cond,
             'conducts': conducts, 'edge_number': edges, 'netparams': jsp_data})
        return cond

    def spaghetti_network_wrapper(self, L, D, VF, junction_res, nsegs, prox, angle, box, boy, boz, is3D=True):

        retries = 1
        key = None
        success = False
        if is3D:
            N = self.calculate_N4VF_cylinders(VF=VF, L=L, D=D, box=box, boy=boy, boz=boz)
        else:
            N = self.calculate_LN_surf(SF=VF, L=L, D=D, box=box, boy=boy)
        PVOL = N * np.pi / 4 * D ** 2 * L
        while not success:
            try:
                key = self.create_key()
                seed = np.random.randint(0, np.iinfo(np.int32).max)
                #                 data_id=self.create_spaghetti_network(key,N=N,L=20,nsegments=2,lengthDev=0,D=D,diamDev=0,proximity=0.0014,primaryAngleDev=180,angleDev=0,box=box,boy=boy,boz=boz,seed=0,air=False,tag='Ag')
                data_id = self.create_spaghetti_network(key, N=N, L=L, nsegments=nsegs, lengthDev=0, D=D, diamDev=0,
                                                        proximity=prox, primaryAngleDev=180, angleDev=angle, box=box,
                                                        boy=boy, boz=boz, seed=seed, air=False, is3D=is3D, tag='Ag')
                #                 data_id=self.create_disk_wire_network(key,box=box,boy=boy,boz=boz,diskN=0,diskL=0,diskD=0.1,wireN=N,wireL=L,wireD=0.13,angleDev=180,proximity=0.0014,steps=0,is3D=True,seed=seed,air=False,tag='Ag')
                network_id = self.analyze(key, data_id)

                success = True
            except Exception as e:

                wait = retries * 5;
                print('Error connecting to server:{} ! Waiting {} secs and re-trying...'.format(self.srv_url, wait))
                print("Exception: ", e)
                time.sleep(wait)
                retries += 1
        self.destroy_key(key)
        cond, conducts = self.get_conductivity_4_network(network_id, junction_res, box, boy, boz, is3D)

        #         #check for edge case
        #         if cond==0 and conducts==1:
        #             jsp_network=jsonpickle.encode(network_id)
        jsp_data = jsonpickle.encode(data_id)
        #             with open('edge_network-{}.jsp'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]),'w') as f:
        #                 f.write(jsp_network)
        #             with open('edge_data-{}.jsp'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]),'w') as f:
        #                 f.write(jsp_data)
        edges = self.get_number_edges(network_id)
        #         self.writer.add_data.remote(self.srv_url,VF,180,angle,cond,conducts,0,jsp_data)
        self.writer.add_data.remote(
            {'server': self.srv_url, 'VF': VF, 'PVOL': PVOL, 'JRES': junction_res, 'N': N, 'nsegs': nsegs, 'prox': prox,
             'primaryAngle': 180, 'secondaryAngle': angle, 'conductivity': cond, 'conducts': conducts,
             'edge_number': edges, 'netparams': jsp_data})
        return cond

    def spaghetti_mix_network_wrapper(self, L1, D1, VF1, angle1, L2, D2, VF2, angle2, box, boy, boz, is3D=True):
        retries = 1
        key = None
        success = False
        if is3D:
            N1 = self.calculate_N4VF_cylinders(VF=VF1, L=L1, D=D1, box=box, boy=boy, boz=boz)
            N2 = self.calculate_N4VF_cylinders(VF=VF2, L=L2, D=D2, box=box, boy=boy, boz=boz)
        else:
            N1 = self.calculate_LN_surf(SF=VF1, L=L1, D=D1, box=box, boy=boy)
            N2 = self.calculate_LN_surf(SF=VF2, L=L2, D=D2, box=box, boy=boy)
        while not success:
            try:
                key = self.create_key()
                seed = np.random.randint(0, np.iinfo(np.int32).max)
                #                 data_id=self.create_spaghetti_network(key,N=N,L=20,nsegments=2,lengthDev=0,D=D,diamDev=0,proximity=0.0014,primaryAngleDev=180,angleDev=0,box=box,boy=boy,boz=boz,seed=0,air=False,tag='Ag')
                data_id = self.create_spaghetti_network(key, N=N1, L=L1, nsegments=5, lengthDev=0, D=D1, diamDev=0,
                                                        proximity=0.0014, primaryAngleDev=180, angleDev=angle1, box=box,
                                                        boy=boy, boz=boz, seed=seed, air=False, is3D=is3D, tag='Ag')
                data_id = self.create_spaghetti_network(key, N=N2, L=L2, nsegments=5, lengthDev=0, D=D2, diamDev=0,
                                                        proximity=0.0014, primaryAngleDev=180, angleDev=angle2, box=box,
                                                        boy=boy, boz=boz, seed=seed, air=False, is3D=is3D, tag='Ag')
                #                 data_id=self.create_disk_wire_network(key,box=box,boy=boy,boz=boz,diskN=0,diskL=0,diskD=0.1,wireN=N,wireL=L,wireD=0.13,angleDev=180,proximity=0.0014,steps=0,is3D=True,seed=seed,air=False,tag='Ag')
                network_id = self.analyze(key, data_id)

                success = True
            except Exception as e:

                wait = retries * 5;
                print('Error connecting to server:{} ! Waiting {} secs and re-trying...'.format(self.srv_url, wait))
                print("Exception: ", e)
                time.sleep(wait)
                retries += 1
        self.destroy_key(key)
        cond, conducts = self.get_conductivity_4_network(network_id,box,boy,boz, is3D)

        #         #check for edge case
        #         if cond==0 and conducts==1:
        #             jsp_network=jsonpickle.encode(network_id)
        jsp_data = jsonpickle.encode(data_id)
        #             with open('edge_network-{}.jsp'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]),'w') as f:
        #                 f.write(jsp_network)
        #             with open('edge_data-{}.jsp'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]),'w') as f:
        #                 f.write(jsp_data)
        edges = self.get_number_edges(network_id)
        #         self.writer.add_data.remote(self.srv_url,VF,180,angle,cond,conducts,0,jsp_data)
        self.writer.add_data.remote(
            {'server': self.srv_url, 'VF': VF1, 'primaryAngle': 180, 'secondaryAngle': angle1, 'conductivity': cond,
             'conducts': conducts, 'edge_number': edges, 'netparams': jsp_data})
        return cond


@ray.remote
class UniversalWriter(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def add_data(self, new_data):
        self.data.append(new_data)
        df = pd.DataFrame(self.data)
        df.to_csv(self.filename, index=False)

    def get_data(self):
        return self.data

    def clear_data(self, filename):
        self.filename = filename
        self.data = []
        return self.filename, len(self.data)


@ray.remote
class CircWrapper(object):
    def __init__(self, lancuda_srv_url):
        self.nf_lancuda = netfitter.NetworkFitter(serverUrl=lancuda_srv_url)

    def get_conductivity(self, circ, box, boy, boz, is3d=True):
        utils = utilities.Utilities(serverUrl=self.nf_lancuda.serverUrl)
        key = self.nf_lancuda.init_steps(circ['circuit'], utils)

        voltage = 100
        drainres = 100
        res = {}
        current_graphs = []

        circ = g2json.modify_integration_time(circ, set_val='5e-5')

        try:
            for n in range(2):
                res[n] = nf_lancuda.make_step(key, X=[voltage], inputids=circ['inputids'], outputids=circ['outputids'],
                                              controlids=[], eq_time=0.0001, utils=utils)
        #             currents=utils.getElementsIVs(key)
        #             gg=g2json.get_currents_for_graph(comb_graph,circ,currents).copy()
        #             current_graphs.append(gg)
        except Exception as e:
            print(e)
            return -99.0;

        nf_lancuda.complete_steps(key, utils)
        res = {0: res}

        current = abs(list(res[0][0].values())[0])
        actual_voltage = abs(voltage - abs(
            current * drainres))  # this is the actual voltage on the device, not whole system, because we always add drain resistance of approx 100Ohm
        #         print("{},{},{},{},{}".format(current,voltage,self.box,self.boy,self.boz))
        try:
            if is3d:
                cond = current / actual_voltage * box / (boy * boz) * 1e-6
            else:
                cond = current / actual_voltage * box / boy
        except Exception as e:
            print(str(e), "actual_voltage:", actual_voltage, "current:", current)
            return 0
        return cond






def main():
    perc = percolator.Percolator(serverUrl="http://landau-nic0.mse.ncsu.edu:15850/percolator/")

    perc_server_pool = [
        "http://landau-nic0.mse.ncsu.edu:15843/percolator/",
        "http://kolmogorov-nic0.mse.ncsu.edu:15843/percolator/",
        "http://spartan.mse.ncsu.edu:15843/percolator/",
        "http://nikolay1.mse.ncsu.edu:2233/percolator/",
        "http://nikolay1.mse.ncsu.edu:2243/percolator/",
        "http://nikolay1.mse.ncsu.edu:2253/percolator/",
        "http://nikolay1.mse.ncsu.edu:2263/percolator/",
        "http://nikolay1.mse.ncsu.edu:2273/percolator/",
        "http://nikolay1.mse.ncsu.edu:2283/percolator/",
        "http://nikolay1.mse.ncsu.edu:2293/percolator/",
        "http://nikolay1.mse.ncsu.edu:2303/percolator/"
    ]

    circ_server_pool = [
        "http://landau-nic0.mse.ncsu.edu:15820/symphony/",
        "http://kolmogorov-nic0.mse.ncsu.edu:15820/symphony/"
    ]

    # writer = UniversalWriter.remote('spaghetti2d_12232019_norelax_noair_prx1p4nm_L20D100_box300_jr500_1.csv')
    writer = UniversalWriter.remote('results/global_scan_1.csv')

    # box, boy, boz = 300, 300, 300
    # input_electrode_arr=perc.create_elect_boxes(elmat=[1,1],plane=0,gap=0.0,box={'x0':-5,'y0':0,'z0':0,'x1':box,'y1':boy,'z1':boz},delta=(5,0,0))
    # output_electrode_arr=perc.create_elect_boxes(elmat=[1,1],plane=1,gap=0.0,box={'x0':0,'y0':0,'z0':0,'x1':box+5,'y1':boy,'z1':boz},delta=(5,0,0))

    circ_workers = []
    for s_u in circ_server_pool * 20:
        circ_workers.append(CircWrapper.remote(lancuda_srv_url=s_u))

    # circ_iterator=itertools.cycle(circ_workers)
    net_workers = []
    for s_u in perc_server_pool * 5:
        net_workers.append(
            NetWorker2.remote(perc_server_url=s_u, nf_lancuda=nf_lancuda, writer=writer, circ_workers=circ_workers))

    # input_params = pd.concat([choice2grid({'VF':np.linspace(0.02,0.10,20),'angles':[0],'jres':[500,2000,5000],'nsegs':[5,10],'L':[20,50,100],'boxmul':[5,10]})]*30,ignore_index=True)
    input_params = pd.concat([choice2grid(
        {'VF': np.linspace(0.02, 0.10, 1), 'angles': [0], 'jres': [500, 2000, 5000], 'nsegs': [5, 10], 'L': [20, 50],
         'boxmul': [5, 10]})] * 1, ignore_index=True)
    input_params = list(input_params.values)
    random.shuffle(input_params)

    net_worker_it=itertools.cycle(net_workers)
    result_ids=[]
    for input_param in input_params:
        VF=input_param[0]
        angle=input_param[1]
        jres=input_param[2]
        nsegs=int(input_param[3])
        L=int(input_param[4])
        box=boy=boz=int(L*input_param[5])
        D=0.10
    #     L=20
    #     w=random.choice(net_workers)
        w=next(net_worker_it)
        cond=w.spaghetti_network_wrapper.remote(L=L,D=D,VF=VF,junction_res=jres,nsegs=nsegs,prox=0.0014,angle=angle,box=box,boy=boy,boz=boz,is3D=False)
    #     cond=w.wire_network_wrapper.remote(L,D,VF,angle,box,boy,boz)
        result_ids.append(cond)
    print("Vuola!!")


if __name__=="__main__":
    main()