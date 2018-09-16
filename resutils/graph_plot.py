import matplotlib
matplotlib.use('QT5Agg')
import networkx as nx
import matplotlib.pyplot as plt
import json


def plot_json_graph(dictdata):
    jsondata  = json.loads(dictdata['circuit'])
    inputids  = dictdata['inputids']
    outputids = dictdata['outputids']

    edgeslist = jsondata

    n=100
    k=3
    p=1.1

    G = nx.watts_strogatz_graph(n=n,k=k,p=p)

    pos = nx.spring_layout(G)



    nx.draw(G, with_labels=True,pos=pos)
    plt.show()

def main():
    return 0

if __name__ == "__main__":
    main()