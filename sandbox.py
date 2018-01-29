import json
from utils import utilities
import time
import networktest
import numpy as np

def main():
    # dat = json.load(open("/home/nifrick/IdeaProjects/CircuitSymphony/src/test/resources/transistor_a.json"))
    # utils = utilities.Utilities(serverUrl="http://localhost:8090/symphony/")
    # response = utils.createNewSimulation()
    # print(response)
    # key = json.loads(response)["key"]

    # response = utils.loadCircuitFromGraphFile(key,"/home/nifrick/PycharmProjects/ResSymphony/results/n100_p0.045_k4_testxor_eqt0_5_date01-14-18-16_03_44_id35.json")
    # print(response)
    # utils.start(key)
    # print(utils.time(key))
    # time.sleep(1.0)
    # print(utils.time(key))
    # utils.kill(key)

    jsonstr = json.load(open("/home/nifrick/PycharmProjects/ResSymphony/n100_p0.045_k4_testxor_eqt0.5_date01-14-18-16_03_44_id35.json",'r'))

    inputcirc={}
    inputcirc['circuit'] = json.dumps(jsonstr)
    inputcirc['inputids'] = [201,202]
    inputcirc['outputids'] = [203,205,207,209,211,213,215,217]

    result=networktest.truthtabletest(type='xor', circuit=inputcirc,eq_time=0.5)


    logreg_results = networktest.logreg_test(result)
    score = np.sum(np.abs(logreg_results))

    print(logreg_results)

    print("Score:",score)

    # result = generate_random_net_circuit(n=50,p=3,nin=2,nout=4)
    # jsonstr=result['circuit']
    # inpuids=result['inputids']
    # outputids=result['outputids']
    # with open('test2.json','w') as json_file:
    #     json_file.write(jsonstr)

if __name__ == "__main__":
    main()