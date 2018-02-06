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

ttables = {}
ttables['xor'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
ttables['or'] = [[-1, -1, 0], [-1, 1, 1], [1, -1, 1], [1, 1, 0]]
ttables['and'] = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

def generate_random_net(n=20,p=2,k=4,net_type='ws'):
    # G = nx.complete_graph(10)
    # G = nx.fast_gnp_random_graph(n=n,p=p)
    if net_type =='ws':
        G = nx.watts_strogatz_graph(n=n, k=k, p=p)
    elif net_type == 'ba':
        G = nx.barabasi_albert_graph(n=n,p=p)

    print(G.edges())
    # nx.draw(G, with_labels=True)
    # plt.show()
    return G

# def generate_random_net_circuit(n=20,p=2):
#     Ron=100.
#     Roff=32000.
#     dopwidth=0.
#     totwidth=1.0E-8
#     mobility=1.0E-10
#
#     G = generate_random_net(n,p)
#     edges = G.edges()
#     doc = {}
#     doc[0]=['$', 1, 5e-06, 10.634267539816555, 43, 2.0, 50]
#     for e,elemid in zip(edges,range(1,len(edges)+1)):
#         # lst=["m",e[0],e[1],0,i,"100.0","32000.0","0.0","1.0E-8","1.0E-10"]
#         lst = ["m", e[0], e[1], 0, elemid, str(Ron), str(Roff), str(dopwidth), str(totwidth), str(mobility)]
#         doc[elemid]=lst
#     return json.dumps(doc,sort_keys=True,indent=4)


def generate_random_net_circuit(n=10,p=2,k=4,nin=2,nout=2,el_type='m',rndmzd=False,net_type='ws'):

    #memristor base configuration
    Ron=100.
    Roff=32000.
    dopwidth=0.
    totwidth=1.0E-8
    mobility=1.0E-10

    drainres = 100

    elemceil = 10000 # maximum id of element

    G = generate_random_net(n=n,p=p,k=k,net_type=net_type)
    edges = G.edges()
    doc = {}
    doc[0]=['$', 1, 5e-06, 10.634267539816555, 43, 2.0, 50]
    for e,elemid in zip(edges,range(1,len(edges)+1)):
        # lst=["m",e[0],e[1],0,i,"100.0","32000.0","0.0","1.0E-8","1.0E-10"]
        if el_type=='m':
            totwidth_rnd=totwidth+random.uniform(-totwidth/5.,totwidth/5.)
            dopwidth_rnd=random.uniform(0.,totwidth_rnd)
            lst = ["m", e[0], e[1], 0, elemid, str(Ron), str(Roff), str(dopwidth if rndmzd else dopwidth_rnd), str(totwidth if rndmzd else totwidth_rnd), str(mobility)]
        elif el_type=='d':
            lst = ["d", e[0], e[1], 1, elemid, "0.805904"]
        doc[elemid]=lst

    nodes=list(G.nodes)

    inoutnodes = random.sample(nodes,nin+nout)

    inputids=[]
    outputids=[]

    for k in inoutnodes[:nin]:
        elemid+=1
        elemceil-=1
        # lst = ["R", k, elemceil, 0, elemid, "2", "40.0", "0.0", "0.0", "0.0", "0.5"]
        lst = ["R", k, elemceil, 0, elemid, "0", "40.0", "0.01", "0.0", "0.0", "0.5"]
        doc[elemid]=lst
        inputids.append(elemid)

    for k in inoutnodes[nin:nin+nout]:
        elemid+=1
        elemceil-=1
        lst = ["r", k, elemceil, 0, elemid, str(drainres)]
        doc[elemid] = lst
        outputids.append(elemid)

        elemid+=1
        elemsav=elemceil
        elemceil-=1
        lst = ["g", elemsav, elemceil, 0, 0]
        doc[elemid]=lst

    result={}
    result['circuit']=json.dumps(doc,sort_keys=True,indent=4)
    result['inputids']=inputids
    result['outputids']=outputids

    return result


def set_inputs(key,utils,ids,vals):
    for id, val in zip(ids,vals):
        response = utils.setElementProperty(key, str(id), "maxVoltage", str(val))

def read_outputs(key,utils,ids):
    result=[]
    for id in ids:
        utils.getCurrent(key, str(id))
    return result;

def truthtabletest(type='xor',circuit='',eq_time=0.5,iterations=1):

    jsonstr=circuit['circuit']
    inputids=circuit['inputids']
    outputids=circuit['outputids']

    utils = utilities.Utilities(serverUrl="http://localhost:8090/symphony/")

    # response = utils.createNewSimulation()
    # print(response)
    # key = json.loads(response)["key"]
    #
    # response = utils.loadCircuitFromGraphString(key, jsonstr)
    # print(response)
    #
    # utils.start(key)

    ttable = list(ttables[type])


    results=[]

    # for item in ttable*iterations:
    #     outvals = ttable_single_test(eq_time, inputids, item, jsonstr, outputids, utils)

    items = ttable*iterations
    with mp.pool.ThreadPool(processes=len(ttable)*iterations) as pool:
        outvals = pool.starmap(ttable_single_test,zip(repeat(eq_time),repeat(inputids),items,repeat(jsonstr),repeat(outputids),repeat(utils)))

    for outval in outvals:
        results.append(outval)

    # response = utils.getElementProperty(key, str(inputids[0]), "maxVoltage")
    # print(response)
    #
    # response = utils.getCurrent(key, str(outputids[0]))
    # print(response)
    #
    # time.sleep(2)
    #
    # response = utils.setElementProperty(key, str(inputids[0]), "maxVoltage", "100.5")
    # print(response)
    #
    # response = utils.getElementProperty(key, str(inputids[0]), "maxVoltage")
    # print(response)
    #
    # time.sleep(2)
    #
    # response = utils.getCurrent(key, str(outputids[0]))
    # print(response)

    # utils.kill(key)

    return results


def ttable_single_test(eq_time, inputids, item, jsonstr, outputids, utils):
    response = utils.createNewSimulation()
    print(response)
    key = json.loads(response)["key"]
    response = utils.loadCircuitFromGraphString(key, jsonstr)
    print(response)
    utils.start(key)
    inoutvals = {}
    print("Setting up inputs: ", item[:-1])
    response = utils.setElementProperty(key, str(inputids[0]), "maxVoltage",
                                        str(item[0] + (np.random.rand() - 0.5) / 3))
    response = utils.setElementProperty(key, str(inputids[1]), "maxVoltage",
                                        str(item[1] + (np.random.rand() - 0.5) / 3))
    print("Waiting to equilibrate: {} secs".format(eq_time))
    time.sleep(eq_time)
    outvals = []
    for outid in outputids:
        response = utils.getCurrent(key, str(outid))
        curval = json.loads(response)['value']
        print("Output current vals: ", curval)
        outvals.append(curval)
    utils.kill(key)
    outvals.append(item[2])
    return outvals


def ttt_launcher(ntests=1,n=30,p=2,k=4,nin=2,nout=5,eq_time=0.5, test_type='xor', save_best=False,iterations=1,el_type='m',rndmzd=False,net_type='ws'):
    # jsonstr = json.dumps(json.load(open("/home/nifrick/PycharmProjects/ResSymphony/resources/test2_final.json")))
    # inputids=[197,198]
    # outputids=[199,201,203,205]
    # input={}
    # input['circuit']=jsonstr
    # input['inputids']=inputids
    # input['outputids']=outputids

    inputcirc = generate_random_net_circuit(n=n,p=p,k=k,nin=nin,nout=nout,el_type=el_type,rndmzd=rndmzd,net_type=net_type)

    # plott.plot_json_graph(inputcirc['circuit'])

    results=[]
    for i in range(ntests):
        result=truthtabletest(type=test_type, circuit=inputcirc,eq_time=eq_time,iterations=iterations)
        for res in result:
            results.append(res)

    if save_best:
        logreg_results = logreg_test(results)
        score = np.sum(np.abs(logreg_results))
        if score < 1.:
            jsonstr = inputcirc['circuit']
            with open('./results/' + 'n{}_p{}_k{}_test{}_eqt{}_date{}_id{}'.format(n, p, k, test_type, eq_time,
                                                                                   time.strftime("%m-%d-%y-%H_%M_%S"),
                                                                                   34) + '.json', 'w') as f:
                f.write(json.dumps(jsonstr, sort_keys=True))

    return results, inputcirc['circuit']

def logreg_test(results):
    x = np.asarray(results)[:,:-1]
    y = np.asarray(results)[:,-1]
    logreg = linear_model.LogisticRegression(C=.5)
    logreg.fit(x,y)

    preddiff = []
    for res in results:
        preddiff.append(res[-1] - logreg.predict([res[:-1]]))

    return preddiff

def minimize_res(n,p,k,eq_time,nout):
    # results = ttt_launcher(ntests=1, n=80, p=0.1, k=4, nin=2, nout=6, eq_time=0.9)
    results,inputcirc = ttt_launcher(ntests=1, n=n, p=p, k=k, nin=2,
                                     nout=nout, eq_time=eq_time,save_best=True,
                                     test_type='xor',iterations=20,
                                     el_type='m',rndmzd=True,net_type='ws')
    print(np.array(results).tolist())
    logreg_results = logreg_test(results)
    plott.plot3d(results,inputcirc)
    return 10 - np.sum(np.abs(logreg_results))

def main():

    param_grid = {'n':[20,30]}

    args = {'k':4,'eq_time':2.5,'p':1.5,'nout':3}
    best_params, best_score, score_results = maximize(minimize_res, param_grid, args, verbose=True)

    print(best_params,best_score,score_results)

    # jsonstr = json.load(open("/home/nifrick/IdeaProjects/CircuitSymphony/src/test/resources/transistor_a.json"))

    # dat = json.load(open("/home/nifrick/IdeaProjects/CircuitSymphony/src/test/resources/transistor_a.json"))
    # utils = utilities.Utilities(serverUrl="http://localhost:8090/symphony/"

    # result = generate_random_net_circuit(n=50,p=3,nin=2,nout=4)
    # jsonstr=result['circuit']
    # inpuids=result['inputids']
    # outputids=result['outputids']
    # with open('test2.json','w') as json_file:
    #     json_file.write(jsonstr)


    # response = utils.createNewSimulation()
    # print(response)
    # key = json.loads(response)["key"]
    #
    # jsonstr = json.dumps(json.load(open("/home/nifrick/PycharmProjects/ResSymphony/resources/delteme.json")))
    # response = utils.loadCircuitFromGraphString(key, jsonstr)
    # print(response)
    #
    # response = utils.getElementProperty(key, "1", "maxVoltage")
    # print(response)
    #
    # response = utils.setElementProperty(key, "1", "maxVoltage", "100.5")
    # print(response)
    #
    # response = utils.getElementProperty(key, "1", "maxVoltage")
    # print(response)






if __name__ == "__main__":
    main()