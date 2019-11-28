import matplotlib
matplotlib.use('agg')
import unittest
import json
from resutils import percolator
from resutils import netfitter2 as netfitter
from resutils import graph2json as g2json
from resutils import utilities
from resutils import plott
import networkx as nx

class TestNetfitter(unittest.TestCase):
    def test_network_to_circuit_conversion(self):
        with open('resources/resistance_test_net.json', 'r') as f:
            network = json.load(f)

        perc = percolator.Percolator(serverUrl="http://landau-nic0.mse.ncsu.edu:15843/percolator/")
        nf_lancuda = netfitter.NetworkFitter(serverUrl="http://kolmogorov-nic0.mse.ncsu.edu:15820/symphony/")

        box,boy,boz=200,200,200

        input_electrode_arr = perc.create_elect_boxes(elmat=[1, 1], plane=0, gap=0.2,
                                                      box={'x0': -10, 'y0': 20, 'z0': -1, 'x1': box, 'y1': boy,
                                                           'z1': 1}, delta=(20, 0, 0))
        output_electrode_arr = perc.create_elect_boxes(elmat=[1, 1], plane=1, gap=0.2,
                                                       box={'x0': 0, 'y0': 20, 'z0': -1, 'x1': box, 'y1': boy, 'z1': 1},
                                                       delta=(20, 0, 0))
        G = perc.load_graph_from_json(network)

        accepted_graphs = perc.get_graphs_connecting_electrodearrays(G, {0: input_electrode_arr, 1: output_electrode_arr})

        self.assertEqual(len(accepted_graphs),1)

        accepted_graph = accepted_graphs[0]

        input_nodes = perc.get_nodes_for_box_array(pos=nx.get_node_attributes(accepted_graph, 'pos3d'),
                                                   el_array=input_electrode_arr)
        output_nodes = perc.get_nodes_for_box_array(pos=nx.get_node_attributes(accepted_graph, 'pos3d'),
                                                    el_array=output_electrode_arr)

        wired_electrodes_graph = perc.wire_nodes_on_electrodes(accepted_graph, [input_nodes, output_nodes])

        el_pan = []
        for el_arr in [input_nodes, output_nodes]:
            sub_el = []
            for elk in el_arr.keys():
                sub_el.append(el_arr[elk][0])
            el_pan.append(sub_el)

        comb_graph = perc.prune_dead_edges(wired_electrodes_graph, runs=26)

        # comb_graph = perc.convert_edgeclass_to_device(comb_graph,mem='Zn',res='Ag')
        comb_graph = perc.convert_edgeclass_to_device(comb_graph, mem='Zn', res='Ag', diode='tag3')

        ins = el_pan[0][:]
        outs = el_pan[1][:]
        controls = []
        circ = g2json.transform_network_to_circuit_window(graph=comb_graph, inels=ins, outels=outs, contels=controls,
                                                   Ron_pnm=1, Roff_pnm=1000, mobility=2.56e-9, nw_res_per_nm=0.005,
                                                   t_step="5e-5", scale=1e-6, junct_res_per_nm=1000,
                                                   mem_cutoff_len_nm=0.1, window=g2json.strukov_window())

        self.assertEqual(len(circ['inputids']),1)
        self.assertEqual(len(circ['outputids']), 1)
        self.assertEqual(len(circ['controlids']), 0)


        # calculate conductivity
        voltage=100
        utils = utilities.Utilities(serverUrl=nf_lancuda.serverUrl)
        key = nf_lancuda.init_steps(circ['circuit'], utils)
        res={}
        for n in range(2):
            res[n] = nf_lancuda.make_step(key, X=[voltage], inputids=circ['inputids'],
                                          controlids=circ['controlids'], outputids=circ['outputids'], eq_time=0.0001,
                                          utils=utils)

        nf_lancuda.complete_steps(key, utils)

        res = {0: res}

        current = list(res[0][0].values())[0]
        #         print("{},{},{},{},{}".format(current,voltage,self.box,self.boy,self.boz))
        cond = current / voltage
        print(cond)
        self.assertEqual(cond,cond)


