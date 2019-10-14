from __future__ import print_function
from resutils import utilities
import resutils.plott
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
from resutils import nxgtutils as ngut
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm_notebook
from tqdm import tqdm

DAT_DELTA = 0.03  # multiplier for voltage variation
logger = logging.getLogger(__name__)


class NetworkFitter():
    def __init__(self, n_jobs=1, circuit='', eq_time=0.5, iterations=1,serverUrl=''):
        self.n_jobs = n_jobs
        self.circuit = circuit
        self.eq_time = eq_time
        self.iterations = iterations

        # read server path
        if serverUrl=='':
            cp = configparser.ConfigParser()
            cp.read('/home/nifrick/PycharmProjects/ressymphony/config/config.ini')
            self.serverUrl = cp.get('ServerConfig', 'serverurl')
        else:
            self.serverUrl = serverUrl

    def init_steps(self,jsonstr,utils):
        response = utils.createNewSimulation()
        logger.debug(response)
        key = json.loads(response)["key"]
        response = utils.loadCircuitFromGraphString(key, jsonstr)
        logger.debug(response)

        return key

    def make_step(self,key,X,inputids,controlids,outputids,eq_time,utils):

        vinids=inputids+controlids
        for inputid, idnum in zip(vinids, range(len(vinids))):
            response = utils.setElementProperty(key, str(inputid), "maxVoltage",
                                                str(X[idnum]))

        # print("Waiting to equilibrate: {} secs".format(eq_time))
        logger.info("Waiting to equilibrate: {} secs".format(eq_time))
        response = utils.startForAndWait(key, eq_time)
        if "Singular".lower() in (json.loads(response)['message']).lower():
            raise ValueError("Singular Matrix");

        utils.stop(key)

        outvals = {}
        # print("Done equilibrating, reading output values")
        logger.info("Done equilibrating, reading output values")
        for outid in outputids:
            response = utils.getCurrent(key, str(outid))
            curval = json.loads(response)['value']
            # print("Output current vals: ", curval)
            outvals[outid]=curval

        return outvals


    def complete_steps(self,key,utils):
        utils.stop(key)
        utils.kill(key)

    def run_single_sim(self, X, y, inputids, outputids, jsonstr, eq_time, utils, perturb=False):
        response = utils.createNewSimulation()

        # print(response)
        logger.debug(response)
        key = json.loads(response)["key"]
        response = utils.loadCircuitFromGraphString(key, jsonstr)
        # print(response)
        logger.debug(response)
        # resutils.start(key)
        inoutvals = {}
        # print("Setting up inputs:", X, "for outputs:",y)
        logger.debug("Setting up inputs: {} for outputs: {} ".format(X, y))

        for inputid, idnum in zip(inputids, range(len(inputids))):
            response = utils.setElementProperty(key, str(inputid), "maxVoltage",
                                                str(X[idnum]))

        # print("Waiting to equilibrate: {} secs".format(eq_time))
        logger.info("Waiting to equilibrate: {} secs".format(eq_time))
        # utils.startForAndWait(key, eq_time)
        response = utils.startForAndWait(key, eq_time)
        if "Singular".lower() in (json.loads(response)['message']).lower():
            raise ValueError("Singular Matrix");
        utils.stop(key)

        outvals = []
        # print("Done equilibrating, reading output values")
        logger.info("Done equilibrating, reading output values")
        for outid in outputids:
            response = utils.getCurrent(key, str(outid))
            curval = json.loads(response)['value']
            # print("Output current vals: ", curval)
            outvals.append(curval)
        utils.kill(key)
        outvals.append(y)
        return outvals

    def run_single_sim_control(self, X, y, control, inputids, outputids, controlids, jsonstr, eq_time, utils, perturb=False):
        response = utils.createNewSimulation()

        # print(response)
        logger.debug(response)
        key = json.loads(response)["key"]
        response = utils.loadCircuitFromGraphString(key, jsonstr)
        # print(response)
        logger.debug(response)
        # resutils.start(key)
        inoutvals = {}
        # print("Setting up inputs:", X, "for outputs:",y)
        logger.debug("Setting up inputs: {} for outputs: {} and control: {}".format(X, y, control))

        for inputid, idnum in zip(inputids, range(len(inputids))):
            response = utils.setElementProperty(key, str(inputid), "maxVoltage",
                                                str(X[idnum]))

        for controlid, idnum in zip(controlids, range(len(controlids))):
            response = utils.setElementProperty(key, str(controlid), "maxVoltage",
                                                str(control[idnum]))

        # print("Waiting to equilibrate: {} secs".format(eq_time))
        logger.info("Waiting to equilibrate: {} secs".format(eq_time))
        # utils.startForAndWait(key, eq_time)
        response = utils.startForAndWait(key, eq_time)
        if "Singular".lower() in (json.loads(response)['message']).lower():
            raise ValueError("Singular Matrix");
        utils.stop(key)

        outvals = []
        # print("Done equilibrating, reading output values")
        logger.info("Done equilibrating, reading output values")
        for outid in outputids:
            response = utils.getCurrent(key, str(outid))
            curval = json.loads(response)['value']
            # print("Output current vals: ", curval)
            outvals.append(curval)
        utils.kill(key)
        outvals.append(y)
        return outvals

    def run_single_sim_series(self, X, y, inputids, outputids, jsonstr, eq_time, utils, repeat=1):
        response = utils.createNewSimulation()
        # print(response)
        logger.debug(response)
        key = json.loads(response)["key"]
        response = utils.loadCircuitFromGraphString(key, jsonstr)
        # print(response)
        logger.debug(response)
        # resutils.stop(key)
        inoutvals = {}
        outvals = {}
        # print("Setting up inputs:", X, "for outputs:",y)
        logger.debug("Setting up inputs: {} for outputs: {} ".format(X, y))
        for k in range(repeat):
            for inputid, idnum in zip(inputids, range(len(inputids))):
                response = utils.setElementProperty(key, str(inputid), "maxVoltage",
                                                    str(X[idnum]))

            # print("Waiting to equilibrate: {} secs".format(eq_time))
            logger.info("Waiting to equilibrate: {} secs".format(eq_time))
            # utils.startForAndWait(key, eq_time)
            response = utils.startForAndWait(key, eq_time)
            if "Singular".lower() in (json.loads(response)['message']).lower():
                raise ValueError("Singular Matrix");
            utils.stop(key)
            # print("Done equilibrating, reading output values")
            logger.info("Done equilibrating, reading output values")
            outvals[k] = {}
            for outid in outputids:
                response = utils.peekCurrent(key, str(outid))
                curval = json.loads(response)['value']
                # print("Output current vals: ", curval)
                outvals[k][outid] = curval

        utils.kill(key)
        # outvals.append(y)
        return outvals

    def run_continuous_sim(self, X, inputids, outputids, jsonstr, eq_time, utils):
        response = utils.createNewSimulation()
        # print(response)
        logger.debug(response)
        key = json.loads(response)["key"]
        response = utils.loadCircuitFromGraphString(key, jsonstr)
        # print(response)
        logger.debug(response)
        # resutils.stop(key)
        inoutvals = {}
        outvals = {}
        # print("Setting up inputs:", X, "for outputs:",y)
        logger.debug("Setting up inputs: {}".format(X))
        for k,X_row in tqdm(enumerate(X)):
            for inputid, idnum in zip(inputids, range(len(inputids))):
                response = utils.setElementProperty(key, str(inputid), "maxVoltage",
                                                    str(X_row[idnum]))

            # print("Waiting to equilibrate: {} secs".format(eq_time))
            logger.info("Waiting to equilibrate: {} secs".format(eq_time))
            # utils.startForAndWait(key, eq_time)
            response = utils.startForAndWait(key, eq_time)
            if "Singular".lower() in (json.loads(response)['message']).lower():
                raise ValueError("Singular Matrix");
            utils.stop(key)
            # print("Done equilibrating, reading output values")
            logger.info("Done equilibrating, reading output values")
            outvals[k] = {}
            for outid in outputids:
                response = utils.peekCurrent(key, str(outid))
                curval = json.loads(response)['value']
                # print("Output current vals: ", curval)
                outvals[k][outid] = curval

        utils.kill(key)
        # outvals.append(y)
        return outvals

    def network_eval(self, X, y, circ="", n_jobs=0):
        if circ != "":
            self.circuit = circ
        jsonstr = self.circuit['circuit']
        inputids = self.circuit['inputids']
        outputids = self.circuit['outputids']

        utils = utilities.Utilities(serverUrl=self.serverUrl)

        results = []

        if n_jobs == 0:
            n_jobs = len(y)

        with mp.pool.ThreadPool(processes=n_jobs) as pool:
            outvals = pool.starmap(self.run_single_sim,
                                   zip(X, y,
                                       repeat(inputids),
                                       repeat(outputids),
                                       repeat(jsonstr),
                                       repeat(self.eq_time),
                                       repeat(utils)))

        for outval in outvals:
            results.append(outval)

        return results

    def network_eval_control(self, X, y, control, circ="", n_jobs=0):
        if circ != "":
            self.circuit = circ
        jsonstr = self.circuit['circuit']
        inputids = self.circuit['inputids']
        outputids = self.circuit['outputids']

        controlids = []
        if len(control) >0:
            controlids = self.circuit['controlids']

        utils = utilities.Utilities(serverUrl=self.serverUrl)

        results = []

        if n_jobs == 0:
            n_jobs = len(y)

        with mp.pool.ThreadPool(processes=n_jobs) as pool:
            outvals = pool.starmap(self.run_single_sim_control,
                                   zip(X, y, control,
                                       repeat(inputids),
                                       repeat(outputids),
                                       repeat(controlids),
                                       repeat(jsonstr),
                                       repeat(self.eq_time),
                                       repeat(utils)))

        for outval in outvals:
            results.append(outval)

        return results

    def logreg_fit(self, X, y, rescale=False):
        X = np.concatenate((X, y.reshape((-1, 1))), axis=1)
        if rescale == True:
            std_scaler = StandardScaler()
            std_scaler.fit(X)
            X = std_scaler.transform(X)
        x = np.asarray(X[:, :-1])
        y = np.asarray(X[:, -1])
        logreg = linear_model.LogisticRegression(C=.5)
        logreg.fit(x, y)

        # preddiff = []
        # for res in X:
        #     preddiff.append(y - logreg.predict([res]))

        return logreg.score(X, y)

    def logreg_fit_mat(self, inmat, rescale=False):
        y = np.array(inmat)[:, -1]
        if rescale:
            std_scaler = StandardScaler()
            std_scaler.fit(inmat)
            inmat = std_scaler.transform(inmat)
        X = np.array(inmat)[:, :-1]
        logreg = linear_model.LogisticRegression(C=300.5, verbose=True, tol=1e-8, fit_intercept=True)
        logreg.fit(X, y)

        # preddiff = []
        # for res in X:
        #     preddiff.append(y - logreg.predict([res]))

        return logreg.score(X, y)

    def generate_random_net(self, n=20, p=2, k=4, rmp=0.1, net_type='ws', plot=False):
        # G = nx.complete_graph(10)
        # G = nx.fast_gnp_random_graph(n=n,p=p)
        if net_type == 'ws':
            G = nx.watts_strogatz_graph(n=n, k=k, p=p)
        elif net_type == 'ba':
            G = nx.barabasi_albert_graph(n=n, p=p)
        elif net_type == 'sq':
            G = ngut.generate_lattice(n=n, dim=2, rmp=rmp, periodic=False)
        elif net_type == 'co':
            G = nx.complete_graph(n=n)

        # print("Total edges generated", len(G.edges()))
        logger.info("Total edges generated" + str(len(G.edges())))
        if plot:
            nx.draw(G, with_labels=True)
            # plt.savefig("graph.png")
            plt.show()
        return G

    def generate_random_net_circuit(self, n=10, p=2, k=4, nin=2, nout=2,ncont=0, el_type='m', rndmzd=False, rmp=0.1, net_type='ws', \
                                    Ron=500,Roff=10000,dopwidth=0,totwidth=1.0E-8,totwidth_rnd_delta=5.,mobility=1.0E-10,drainres=100,t_step="5e-6"):

        # memristor base configuration
        Ron = Ron
        Roff = Roff
        dopwidth = dopwidth
        totwidth = totwidth
        mobility = mobility

        drainres = drainres

        elemceil = 100000  # maximum id of element

        G = self.generate_random_net(n=n, p=p, k=k, rmp=rmp, net_type=net_type)
        edges = G.edges()
        doc = {}
        doc[0] = ['$', 1, t_step, 10.634267539816555, 43, 2.0, 50]
        for elemid, ed in enumerate(edges, 1):
            # lst=["m",e[0],e[1],0,i,"100.0","32000.0","0.0","1.0E-8","1.0E-10"]
            if el_type == 'm':
                totwidth_rnd = totwidth + np.max([0,random.uniform(-totwidth / totwidth_rnd_delta, totwidth / totwidth_rnd_delta)])
                dopwidth_rnd = random.uniform(0., totwidth_rnd)
                lst = ["m", ed[0], ed[1], 0, elemid, str(Ron), str(Roff), str(dopwidth_rnd if rndmzd else dopwidth),
                       str(totwidth_rnd if rndmzd else totwidth), str(mobility)]
            elif el_type == 'd':
                lst = ["d", ed[0], ed[1], 1, elemid, "0.805904"]
            doc[elemid] = lst

        nodes = list(G.nodes)

        inoutcontnodes = random.sample(nodes, nin + nout + ncont)

        inputids = []
        outputids = []
        controlids = []

        for k in inoutcontnodes[:nin]:
            elemid += 1
            elemceil -= 1
            # lst = ["R", k, elemceil, 0, elemid, "2", "40.0", "0.0", "0.0", "0.0", "0.5"]
            lst = ["R", k, elemceil, 0, elemid, "0", "40.0", "0.01", "0.0", "0.0", "0.5"]
            doc[elemid] = lst
            inputids.append(elemid)

        for k in inoutcontnodes[nin:nin + nout]:
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

        for k in inoutcontnodes[nin+nout:nin + nout + ncont]:
            elemid += 1
            elemceil -= 1
            # lst = ["R", k, elemceil, 0, elemid, "2", "40.0", "0.0", "0.0", "0.0", "0.5"]
            lst = ["R", k, elemceil, 0, elemid, "0", "40.0", "0.01", "0.0", "0.0", "0.5"]
            doc[elemid] = lst
            controlids.append(elemid)

        result = {}
        result['circuit'] = json.dumps(doc)
        result['inputids'] = inputids
        result['outputids'] = outputids
        result['controlids'] = controlids

        return result, G


def perturb_X(X, boost=3, var=1):
    Y = X.copy()
    for idx, x in np.ndenumerate(Y):
        Y[idx] = x * boost + (np.random.rand() - 0.5) * var
    #     result = np.array(list(map(lambda t: boost*t + (np.random.rand() - 0.5) * var, X)))
    return Y


def xor_test():
    ttables = {}
    ttables['xor'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
    ttables['or'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
    ttables['and'] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

    # jsonstr = json.dumps(json.load(open("/home/nifrick/PycharmProjects/ressymphony/resources/test2_final.json")))
    # inputids=[197,198]
    # outputids=[199,201,203,205]
    # input={}
    # input['circuit']=jsonstr
    # input['inputids']=inputids
    # input['outputids']=outputids

    ## input from file with inputs and outputs definitions
    # jsonstr = json.load(open("/home/nifrick/PycharmProjects/ressymphony/n100_p0.045_k4_testxor_eqt0.5_date01-14-18-16_03_44_id35.json",'r'))
    #
    # input={}
    # input['circuit'] = json.dumps(jsonstr)
    # input['inputids'] = [201,202]
    # input['outputids'] = [203,205,207,209,211,213,215,217]

    nf = NetworkFitter()
    utils = utilities.Utilities(serverUrl="http://10.152.17.144:8090/symphony/")#nf.serverUrl)

    circ, g = nf.generate_random_net_circuit(n=50, nin=2, nout=3)

    nf.circuit = circ

    data = np.array(ttables['xor'] * 16)
    X = data[:, :-1]
    X = perturb_X(X, boost=20, var=2)
    y = data[:, -1]
    # plott.plot_json_graph(circ['circuit'])

    key = nf.init_steps(circ['circuit'], utils)

    out1 = nf.make_step(key, X=[1,2], inputids=circ['inputids'], outputids=circ['outputids'],controlids=[], eq_time=0.0001, utils=utils)
    out2 = nf.make_step(key, X=[1, 2], inputids=circ['inputids'], outputids=circ['outputids'], controlids=[],
                        eq_time=0.0001, utils=utils)
    out3 = nf.make_step(key, X=[1, 2], inputids=circ['inputids'], outputids=circ['outputids'], controlids=[],
                        eq_time=0.0001, utils=utils)
    out4 = nf.make_step(key, X=[1, 2], inputids=circ['inputids'], outputids=circ['outputids'], controlids=[],
                        eq_time=0.0001, utils=utils)

    # out1 = nf.make_step(key, [1, 2], 0, circ['inputids'], circ['outputids'], 0.0001, utils)
    # out2 = nf.make_step(key, [1, 2], 0, circ['inputids'], circ['outputids'], 0.0001, utils)
    # out3 = nf.make_step(key, [1, 2], 0, circ['inputids'], circ['outputids'], 0.0001, utils)
    nf.complete_steps(key, utils)

    # start = time.time()
    # nf.eq_time = 0.004
    # resx = nf.network_eval(X, y)
    # resutils.plott.plot3d(resx, circ['circuit'])
    # results = nf.logreg_fit(resx, y)
    # end = time.time() - start
    #
    # print("Final result vector: ", np.sum(np.abs(results)))
    # print("Circuit size: ", len(json.loads(circ['circuit']).keys()))
    # print("Simulation time: ", end)
    # return results

def singularity_test():
    y = [1, 1]

    circ=json.load(open("/home/nifrick/PycharmProjects/ressymphony/resources/singular.json",'r'))

    nf_lancuda = NetworkFitter(serverUrl="http://10.152.17.144:8090/symphony/")

    utils = utilities.Utilities(serverUrl=nf_lancuda.serverUrl)
    key = nf_lancuda.init_steps(circ['circuit'], utils)
    res = {}

    try:
        for yval, n in tqdm_notebook(zip(y, range(len(y)))):
            res[n] = nf_lancuda.make_step(key, X=[yval * 10000], inputids=circ['inputids'], outputids=circ['outputids'],
                                          controlids=[], eq_time=0.0001, utils=utils)
    except Exception as e:
        print(e)
        nf_lancuda.complete_steps(key, utils)

    nf_lancuda.complete_steps(key, utils)
    res = {0: res}

    print(res)

def main():
    singularity_test()
    # xor_test()

def other_main():
    nf = NetworkFitter()
    nf.circuit = nf.generate_random_net()
    print(nf.circuit)


if __name__ == "__main__":
    # other_main()
    main()
