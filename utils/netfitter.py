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

if __name__ == "__main__":
    main()