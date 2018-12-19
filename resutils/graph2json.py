import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import resutils.nxgtutils as nxutils


def transform_network_to_circuit(graph, inels, outels, mobility = 2.56E-9, nw_res_per_nm=0.005, t_step="5e-6", scale=1e-9,elemceil = 10000,randomized_mem_width=False):
    pos3d = nx.get_node_attributes(graph, 'pos3d')
    #     el_type='m'
    rndmzd = randomized_mem_width
    # memristor base configuration

    #     Ron = 500.
    #     Roff = 100000.
    #     totwidth = 1.0E-8
    #     dopwidth = 0.5*totwidth


    drainres = 100

    elemceil = elemceil  # maximum id of element

    edges = graph.edges()
    elemtypes = nx.get_edge_attributes(graph, 'edgetype')
    doc = {}
    doc[0] = ['$', 1, t_step, 10.634267539816555, 43, 2.0, 50]

    for elemid, e in enumerate(edges, 1):

        # lst=["m",e[0],e[1],0,i,"100.0","32000.0","0.0","1.0E-8","1.0E-10"]
        p1 = np.array(pos3d[e[0]])
        p2 = np.array(pos3d[e[1]])
        dist = np.linalg.norm(p1 - p2) * scale * 1e9
        totwidth = dist * 1e-9
        dopwidth = dist * 0.5 * 1e-9
        Ron = 100 * dist
        Roff = 1000 * dist
        try:
            el_type = elemtypes[e]
        except:
            el_type = 'm'
            print("Error occured")
            pass
        if el_type == 'm':
            totwidth_rnd = totwidth + random.uniform(-totwidth / 5., totwidth / 5.)
            dopwidth_rnd = random.uniform(0., totwidth_rnd)
            lst = ["m", e[0], e[1], 0, elemid, str(Ron), str(Roff), str(dopwidth if rndmzd else dopwidth_rnd),
                   str(totwidth if rndmzd else totwidth_rnd), str(mobility)]
        elif el_type == 'r':
            lst = ['r', e[0], e[1], 0, elemid, str(dist*nw_res_per_nm)]
        elif el_type == 'd':
            lst = ["d", e[0], e[1], 1, elemid, "0.805904"]
        doc[elemid] = lst

    # nodes = list(G.nodes)

    #     inoutnodes = random.sample(nodes, nin + nout)

    inputids = []
    outputids = []

    for node in inels:
        elemid += 1
        elemceil -= 1
        # lst = ["R", k, elemceil, 0, elemid, "2", "40.0", "0.0", "0.0", "0.0", "0.5"]1
        #         idk=random.choice(inels[k])
        idk = node
        lst = ["R", idk, elemceil, 0, elemid, "0", "40.0", "0.0", "0.0", "0.0", "0.5"]
        doc[elemid] = lst
        inputids.append(elemid)

    for node in outels:
        elemid += 1
        elemceil -= 1
        #         idk=random.choice(outels[k])
        idk = node
        lst = ["r", idk, elemceil, 0, elemid, str(drainres)]
        doc[elemid] = lst
        outputids.append(elemid)

        elemid += 1
        elemsav = elemceil
        elemceil -= 1
        lst = ["g", elemsav, elemceil, 0, 0]
        doc[elemid] = lst

    result = {}
    result['circuit'] = json.dumps(doc)
    result['inputids'] = [f for f in inputids]
    result['outputids'] = [f for f in outputids]

    return result

    # short connections within electrode boundary


#     for k in inels.keys():
#         wires= list(itertools.combinations(inels[k],2))
#         for wire in wires:
def transform_network_to_circuit_plain(graph, inels, outels, t_step="5e-6", scale=1e-9):
    pos3d = nx.get_node_attributes(graph, 'pos3d')
    rndmzd = False
    Ron = 5000.
    Roff = 1000000.
    # memristor base configuration

    dopwidth = 0.
    totwidth = 1.0E-6
    mobility = 2.56E-9

    drainres = 100

    elemceil = 10000  # maximum id of element

    edges = graph.edges()
    elemtypes = nx.get_edge_attributes(graph, 'edgetype')
    doc = {}
    doc[0] = ['$', 1, t_step, 10.634267539816555, 43, 2.0, 50]

    for elemid, e in enumerate(edges, 1):

        # lst=["m",e[0],e[1],0,i,"100.0","32000.0","0.0","1.0E-8","1.0E-10"]
        p1 = np.array(pos3d[e[0]])
        p2 = np.array(pos3d[e[1]])
        dist = np.linalg.norm(p1 - p2) * scale * 1e9
        #         totwidth = dist * 1e-9
        #         dopwidth = dist * 0.5 * 1e-9
        #         Ron = 100 * dist
        #         Roff = 1000 * dist
        el_type = elemtypes[e]
        #         try:
        #             el_type = elemtypes[e]
        #         except:
        #             el_type = 'm'
        #             pass
        if el_type == 'm':
            totwidth_rnd = totwidth + random.uniform(-totwidth / 5., totwidth / 5.)
            dopwidth_rnd = random.uniform(0., totwidth_rnd)
            lst = ["m", e[0], e[1], 0, elemid, str(Ron), str(Roff), str(dopwidth if rndmzd else dopwidth_rnd),
                   str(totwidth if rndmzd else totwidth_rnd), str(mobility)]
        elif el_type == 'r':
            lst = ['r', e[0], e[1], 0, elemid, str(Ron)]
        elif el_type == 'd':
            lst = ["d", e[0], e[1], 1, elemid, "0.805904"]
        doc[elemid] = lst

    # nodes = list(G.nodes)

    #     inoutnodes = random.sample(nodes, nin + nout)

    inputids = []
    outputids = []

    for node in inels:
        elemid += 1
        elemceil -= 1
        # lst = ["R", k, elemceil, 0, elemid, "2", "40.0", "0.0", "0.0", "0.0", "0.5"]1
        #         idk=random.choice(inels[k])
        idk = node
        lst = ["R", elemceil, idk, 0, elemid, "0", "40.0", "0.0", "0.0", "0.0", "0.5"]
        doc[elemid] = lst
        inputids.append(elemid)

    for node in outels:
        elemid += 1
        elemceil -= 1
        #         idk=random.choice(outels[k])
        idk = node
        lst = ["r", elemceil, idk, 0, elemid, str(drainres)]
        doc[elemid] = lst
        outputids.append(elemid)

        elemid += 1
        elemsav = elemceil
        elemceil -= 1
        lst = ["g", elemsav, elemceil, 0, 0]
        doc[elemid] = lst

    result = {}
    result['circuit'] = json.dumps(doc)
    result['inputids'] = [f for f in inputids]
    result['outputids'] = [f for f in outputids]

    return result


def generate_random_net_circuit(n=10, p=2, k=4, nin=2, nout=2, el_type='m', rndmzd=False, net_type='ws', t_step="5e-6"):
    # memristor base configuration
    Ron = 5000000.
    Roff = 1000000000.
    dopwidth = 0.
    totwidth = 1.0E-6
    mobility = 2.56E-9

    drainres = 100

    elemceil = 10000  # maximum id of element

    G = generate_random_net(n=n, p=p, k=k, net_type=net_type)
    edges = G.edges()
    doc = {}
    doc[0] = ['$', 1, t_step, 10.634267539816555, 43, 2.0, 50]
    for elemid, ed in enumerate(edges, 1):
        # lst=["m",e[0],e[1],0,i,"100.0","32000.0","0.0","1.0E-8","1.0E-10"]
        if el_type == 'm':
            totwidth_rnd = totwidth + random.uniform(-totwidth / 5., totwidth / 5.)
            dopwidth_rnd = random.uniform(0., totwidth_rnd)
            lst = ["m", ed[0], ed[1], 0, elemid, str(Ron), str(Roff), str(dopwidth if rndmzd else dopwidth_rnd),
                   str(totwidth if rndmzd else totwidth_rnd), str(mobility)]
        elif el_type == 'd':
            lst = ["d", ed[0], ed[1], 1, elemid, "0.805904"]
        doc[elemid] = lst

    nodes = list(G.nodes)

    inoutnodes = random.sample(nodes, nin + nout)

    inputids = []
    outputids = []

    for k in inoutnodes[:nin]:
        elemid += 1
        elemceil -= 1
        # lst = ["R", k, elemceil, 0, elemid, "2", "40.0", "0.0", "0.0", "0.0", "0.5"]
        lst = ["R", k, elemceil, 0, elemid, "0", "40.0", "0.01", "0.0", "0.0", "0.5"]
        doc[elemid] = lst
        inputids.append(elemid)

    for k in inoutnodes[nin:nin + nout]:
        elemid += 1
        elemceil -= 1
        lst = ["r", k, elemceil, 0, elemid, str(drainres)]
        doc[elemid] = lst
        outputids.append(elemid)

        elemid += 1
        elemsav = elemceil
        elemceil -= 1
        lst = ["g", elemsav, elemceil, 0, 0]
        doc[elemid] = lst

    result = {}
    result['circuit'] = json.dumps(doc)
    result['inputids'] = inputids
    result['outputids'] = outputids

    return result, G


def generate_random_net(n=20, p=2, k=4, net_type='ws'):
    # G = nx.complete_graph(10)
    # G = nx.fast_gnp_random_graph(n=n,p=p)
    if net_type == 'ws':
        G = nx.watts_strogatz_graph(n=n, k=k, p=p)
    elif net_type == 'ba':
        G = nx.barabasi_albert_graph(n=n, p=p)
    elif net_type == 'sq':
        #         G = nx.grid_graph([n,n],periodic=False)
        G = nxutils.generate_lattice(n=n, dim=2, rmp=0.1, periodic=False)

    print("Total edges generated", len(G.edges()))
    plt.figure()
    nx.draw(G, with_labels=True)
    # plt.savefig("graph.png")
    plt.show()
    return G


def modify_integration_time(circ, set_val='1e-7'):
    newres = circ
    newcirc = json.loads(circ['circuit'])
    newcirc['0'][2] = set_val
    newres['circuit'] = json.dumps(newcirc)
    return newres


# def batch_plot_single_sim(res, title="", num_elects=3):
#     # plt.subplot(1,2,1)
#     xs = []
#     plt.figure()
#     for meas in res.values():
#         #     res=resharv['1e-6']
#         #     for n in meas.keys():
#
#         #         x1=[meas[k][list(meas[k].keys())[0]] for k in meas.keys()]
#         #         x2=[meas[k][list(meas[k].keys())[1]] for k in meas.keys()]
#         #         x3=[meas[k][list(meas[k].keys())[2]] for k in meas.keys()]
#         #     x4=[meas[k][list(meas[k].keys())[3]] for k in meas.keys()]
#         plt.ticklabel_format(useOffset=False)
#         for n in range(num_elects):
#             x = [meas[k][list(meas[k].keys())[n]] for k in meas.keys()]
#             plt.plot(x, label=str(list(meas[0].keys())[n]))
#         #         plt.plot(x2,label=str(list(meas[0].keys())[1]))
#         #         plt.plot(x3,label=str(list(meas[0].keys())[2]))
#         plt.legend()
#         plt.title(title)
#     #     plt.plot(x4,label='x4')
#     plt.show()
#     #     plt.legend()
#     # plt.show()

def batch_plot_single_sim(res, title=""):
    # plt.subplot(1,2,1)
    num_elects=len(res[0][0].keys())
    xs = []
    plt.figure()
    plt.ticklabel_format(useOffset=False)
    for meas in res.values():
        for n in range(num_elects):
            x = [meas[k][list(meas[k].keys())[n]] for k in meas.keys()]
            plt.plot(x, label=str(list(meas[0].keys())[n]))
    plt.title(title)
    plt.legend()
    plt.show()
