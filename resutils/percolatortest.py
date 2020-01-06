from resutils import percolator
import unittest
import json
import networkx as nx

# class TestPercolator(unittest.TestCase):
#     def test_create(self):
#         perc = percolator.Percolator(serverUrl="http://152.14.71.96:15849/percolator/")
#
#         self.assertEqual(len(json.loads(perc.create()).items()),1)

perc = percolator.Percolator(serverUrl="http://spartan.mse.ncsu.edu:8096/percolator/")

def genarate_2dperc():

    datac = perc.get_default_config()
    key1 = perc.create()
    print(key1)
    datac['cylinder']['enabled'] = True
    datac['cylinder']['diameter'] = 0.1
    datac['cylinder']['length'] = 60
    datac['cylinder']['number'] = 1000
    datac['cylinder']['angleDev'] = 180
    datac['simulation']['boxDimensionX'] = 500
    datac['simulation']['boxDimensionY'] = 500
    datac['simulation']['boxDimensionZ'] = 500
    datac['simulation']['steps'] = 0
    # datac['simulation']['boxDimension']['x']=100
    # datac['simulation']['boxDimension']['y']=100
    # datac['simulation']['boxDimension']['z']=100
    # datac['simulation']['boxPosition']['x']=0
    # datac['simulation']['boxPosition']['y']=0
    # datac['simulation']['boxPosition']['z']=0
    datac['simulation']['proximity'] = 0
    datac['simulation']['is3D'] = 2
    datac['simulation']['seed'] = 2
    datac['simulation']['withAir'] = True

    network = perc.generate_net(key1, **datac)

    G = perc.load_graph_from_json(network)
    nx.get_node_attributes(G, 'pos3d')

def main():

    # genarate_2dperc()
    # perc = percolator.Percolator(serverUrl="http://152.14.71.96:15850/percolator/")
    # perc = percolator.Percolator(serverUrl="http://landau-nic0.mse.ncsu.edu:15846/percolator/")

    datac=perc.get_default_config()
    key1=perc.create()
    print(key1)
    datac['cylinder']['enabled']=True
    datac['cylinder']['diameter'] = 2
    datac['cylinder']['length'] = 30
    datac['cylinder']['number']=100
    datac['simulation']['boxDimensionX']=100
    datac['simulation']['boxDimensionY'] = 100
    datac['simulation']['boxDimensionZ'] = 100
    datac['simulation']['steps']=0
    # datac['simulation']['boxDimension']['x']=100
    # datac['simulation']['boxDimension']['y']=100
    # datac['simulation']['boxDimension']['z']=100
    # datac['simulation']['boxPosition']['x']=0
    # datac['simulation']['boxPosition']['y']=0
    # datac['simulation']['boxPosition']['z']=0
    datac['simulation']['proximity'] = 2
    datac['simulation']['seed'] = 2
    datac['simulation']['withAir'] = True

    print(perc.generate_net(key1,**datac))

    result = perc.generate_network(key=key1, **datac)
    print(result)
    result = perc.analyze(key=key1, withAir=datac['simulation']['withAir'])
    print(result)
    result = perc.export_network(key1)
    print(result)
    result = perc.export_scene(key1)
    print(result)
    result = perc.get_settings(key1)
    print(result)
    result = perc.delete(key1)

if __name__=="__main__":
    main()
