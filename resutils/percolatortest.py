from resutils import percolator
import unittest
import json

# class TestPercolator(unittest.TestCase):
#     def test_create(self):
#         perc = percolator.Percolator(serverUrl="http://152.14.71.96:15849/percolator/")
#
#         self.assertEqual(len(json.loads(perc.create()).items()),1)

def main():
    perc = percolator.Percolator(serverUrl="http://152.14.71.96:15849/percolator/")

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

if __name__=="__main__":
    main()