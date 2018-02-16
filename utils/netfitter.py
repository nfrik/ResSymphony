from __future__ import print_function
from utils import utilities
from utils import plott
import networkx as nx
import matplotlib.pyplot as plt
import json
import random
import time
import numpy as np
from sklearn import linear_model
from evolutionary_search import maximize
from pprint import pprint
import multiprocessing as mp
from itertools import repeat
import configparser

class NetworkFitter():
    def __init__(self, n_jobs=1,circuit='',eq_time=0.5,iterations=1):
        self.n_jobs = n_jobs
        self.circuit=circuit
        self.eq_time=eq_time
        self.iterations=iterations


        #read server path
        cp = configparser.ConfigParser()
        cp.read('../config/config.ini')
        self.serverUrl = cp.get('ServerConfig','serverUrl')

    def run_single_sim(self, eq_time, inputids, X,y, jsonstr, outputids, utils):
        response = utils.createNewSimulation()
        print(response)
        key = json.loads(response)["key"]
        response = utils.loadCircuitFromGraphString(key, jsonstr)
        print(response)
        utils.start(key)
        inoutvals = {}
        print("Setting up inputs: ", X)

        for inputid, idnum in zip(inputids,range(len(inputids))):
            response = utils.setElementProperty(key, str(inputid), "maxVoltage",
                                                str(X[idnum] + (np.random.rand() - 0.5) / 3))
            # response = utils.setElementProperty(key, str(inputids[1]), "maxVoltage",
            #                                     str(item[1] + (np.random.rand() - 0.5) / 3))
        print("Waiting to equilibrate: {} secs".format(eq_time))
        time.sleep(eq_time)
        outvals = []
        for outid in outputids:
            response = utils.getCurrent(key, str(outid))
            curval = json.loads(response)['value']
            print("Output current vals: ", curval)
            outvals.append(curval)
        utils.kill(key)
        outvals.append(y)
        return outvals

    def network_eval(self, X, y):
        jsonstr = self.circuit['circuit']
        inputids = self.circuit['inputids']
        outputids = self.circuit['outputids']

        utils = utilities.Utilities(serverUrl=self.serverUrl)

        results = []

        with mp.pool.ThreadPool(processes=len(y)) as pool:
            outvals = pool.starmap(self.run_single_sim,
                                   zip(repeat(self.eq_time), repeat(inputids), X,y, repeat(jsonstr), repeat(outputids),
                                       repeat(utils)))

        for outval in outvals:
            results.append(outval)


        return results

    def logreg_fit(self,results):
        x = np.asarray(results)[:, :-1]
        y = np.asarray(results)[:, -1]
        logreg = linear_model.LogisticRegression(C=.5)
        logreg.fit(x, y)

        preddiff = []
        for res in results:
            preddiff.append(res[-1] - logreg.predict([res[:-1]]))

        return preddiff

    def generate_random_net(self, n=20, p=2, k=4, net_type='ws'):
        # G = nx.complete_graph(10)
        # G = nx.fast_gnp_random_graph(n=n,p=p)
        if net_type == 'ws':
            G = nx.watts_strogatz_graph(n=n, k=k, p=p)
        elif net_type == 'ba':
            G = nx.barabasi_albert_graph(n=n, p=p)

        print(G.edges())
        # nx.draw(G, with_labels=True)
        # plt.show()
        return G

    def generate_random_net_circuit(self,n=10, p=2, k=4, nin=2, nout=2, el_type='m', rndmzd=False, net_type='ws'):

        # memristor base configuration
        Ron = 100.
        Roff = 32000.
        dopwidth = 0.
        totwidth = 1.0E-8
        mobility = 1.0E-10

        drainres = 100

        elemceil = 10000  # maximum id of element

        G = self.generate_random_net(n=n, p=p, k=k, net_type=net_type)
        edges = G.edges()
        doc = {}
        doc[0] = ['$', 1, 5e-06, 10.634267539816555, 43, 2.0, 50]
        for e, elemid in zip(edges, range(1, len(edges) + 1)):
            # lst=["m",e[0],e[1],0,i,"100.0","32000.0","0.0","1.0E-8","1.0E-10"]
            if el_type == 'm':
                totwidth_rnd = totwidth + random.uniform(-totwidth / 5., totwidth / 5.)
                dopwidth_rnd = random.uniform(0., totwidth_rnd)
                lst = ["m", e[0], e[1], 0, elemid, str(Ron), str(Roff), str(dopwidth if rndmzd else dopwidth_rnd),
                       str(totwidth if rndmzd else totwidth_rnd), str(mobility)]
            elif el_type == 'd':
                lst = ["d", e[0], e[1], 1, elemid, "0.805904"]
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
        result['circuit'] = json.dumps(doc, sort_keys=True, indent=4)
        result['inputids'] = inputids
        result['outputids'] = outputids

        return result

def main():
    ttables = {}
    ttables['xor'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
    ttables['or'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
    ttables['and'] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

    # jsonstr = json.dumps(json.load(open("/home/nifrick/PycharmProjects/ResSymphony/resources/test2_final.json")))
    # inputids=[197,198]
    # outputids=[199,201,203,205]
    # input={}
    # input['circuit']=jsonstr
    # input['inputids']=inputids
    # input['outputids']=outputids

    jsonstr = json.load(open("/home/nifrick/PycharmProjects/ResSymphony/n100_p0.045_k4_testxor_eqt0.5_date01-14-18-16_03_44_id35.json",'r'))

    input={}
    input['circuit'] = json.dumps(jsonstr)
    input['inputids'] = [201,202]
    input['outputids'] = [203,205,207,209,211,213,215,217]


    nf = NetworkFitter(circuit=input)

    data=np.array(ttables['xor']*1)
    X = data[:,:-1]
    y = data[:,-1]

    results = nf.network_eval(X, y)
    results = nf.logreg_fit(results)

    print("Final result vector: ",np.sum(np.abs(results)))
    return results

def other_main():
    nf = NetworkFitter()
    nf.circuit=nf.generate_random_net()
    print(nf.circuit)

if __name__ == "__main__":
    other_main()