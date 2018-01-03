import utilities
import networkx as nx
# import matplotlib.pyplot as plt
import pylab as plt
import json
import pprint

def generate_random_net(n=20,p=2):
    # G = nx.complete_graph(10)
    # G = nx.fast_gnp_random_graph(100,0.3)
    G = nx.barabasi_albert_graph(n,p)
    print(G.edges())
    nx.draw(G, with_labels=True)
    plt.show()
    return G

def generate_random_net_circuit(n=20,p=2):
    Ron=100.
    Roff=32000.
    dopwidth=0.
    totwidth=1.0E-8
    mobility=1.0E-10

    G = generate_random_net(n,p)
    edges = G.edges()
    doc = {}
    doc[0]=['$', 1, 5e-06, 10.634267539816555, 43, 2.0, 50]
    for e,elemid in zip(edges,range(1,len(edges)+1)):
        # lst=["m",e[0],e[1],0,i,"100.0","32000.0","0.0","1.0E-8","1.0E-10"]
        lst = ["m", e[0], e[1], 0, elemid, str(Ron), str(Roff), str(dopwidth), str(totwidth), str(mobility)]
        doc[elemid]=lst
    return json.dumps(doc,sort_keys=True,indent=4)


def main():
    dat = json.load(open("/home/nifrick/IdeaProjects/CircuitSymphony/src/test/resources/transistor_a.json"))
    utils = utilities.Utilities(serverUrl="http://localhost:8090/symphony/")
    # generate_random_net()
    jsonstr=generate_random_net_circuit()

    response = utils.createNewSimulation()
    print(response)
    key = json.loads(response)["key"]


    jsonstr = json.dumps(json.load(open("/home/nifrick/PycharmProjects/ResSymphony/delteme.json")))
    response = utils.loadCircuitFromGraphString(key, jsonstr)
    print(response)

    response = utils.getElementProperty(key, "1", "maxVoltage")
    print(response)

    response = utils.setElementProperty(key, "1", "maxVoltage", "100.5")
    print(response)

    response = utils.getElementProperty(key, "1", "maxVoltage")
    print(response)

    # with open('test.json','w') as json_file:
    #     json_file.write(jsonstr)



if __name__ == "__main__":
    main()