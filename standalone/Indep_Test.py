import json
import requests
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import time

class Indep_test:

    serverUrl = ""

    def __init__(self, serverUrl):
        Indep_test.serverUrl=serverUrl

    def createNewSimulation(self):
        payload = ""
        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        url = Indep_test.serverUrl + "simulations"

        payload = ""

        response = requests.request("POST", url, data=payload, headers=headers)

        return json.loads(json.dumps(response.text))

    # Load circuit represented by graph from uploaded JSON file
    def loadCircuitFromGraphString(self, key, jsonString):
        files = {'file': ('circuit.json', jsonString)}

        url = Indep_test.serverUrl + "simulations/" + key + "/loadCircuitFromGraph"

        response = requests.request("POST", url, files=files)

        return json.loads(json.dumps(response.text))

    def startForAndWait(self,key,seconds):
        url = Indep_test.serverUrl + "simulations/" + key + "/startForAndWait"
        querystring = {"seconds": seconds}

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers, params=querystring)

        return json.loads(json.dumps(response.text))

    # Starts simulation for specified period of simulated circuit seconds and returns immediately
    def startFor(self, key, seconds):
        url = Indep_test.serverUrl + "simulations/" + key + "/start"
        querystring = {"seconds": seconds}

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers, params=querystring)

        return json.loads(json.dumps(response.text))

    def kill(self,key):
        url = Indep_test.serverUrl + "simulations/" + key + "/kill"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers)

        return json.loads(json.dumps(response.text))

    # Changes specified element property to a given value
    def setElementProperty(self,key,elementId, propertyKey, newValue):
        url = Indep_test.serverUrl + "simulations/" + key + "/element/" + elementId+"/property"
        querystring = {"propertyKey": propertyKey,"newValue":newValue}

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.patch(url, headers=headers, params=querystring)

        return json.loads(json.dumps(response.text))

    # Returns simulation time when last measurement was performed
    def peekTime(self, key):
        url = Indep_test.serverUrl + "simulations/" + key + "/peekTime"
        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.request("POST", url, headers=headers)

        return json.loads(json.dumps(response.text))

    # Returns simulation time when last measurement was performed
    def time(self, key):
        url = Indep_test.serverUrl + "simulations/" + key + "/time"
        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.request("POST", url, headers=headers)

        return json.loads(json.dumps(response.text))


def atomic_sim(n):

    utils = Indep_test(serverUrl="http://localhost:8090/symphony/");
    response = utils.createNewSimulation()
    key = json.loads(response)["key"]

    jsoncontent = json.dumps(json.load(open("/home/nifrick/PycharmProjects/ResSymphony/nonlinear_memrist_test.json")))

    inputids = [201, 202]
    outputids = [203, 205, 207, 209, 211, 213, 215, 217]

    response = utils.loadCircuitFromGraphString(key, jsoncontent)
    utils.setElementProperty(key, str(inputids[0]), "maxVoltage",str(np.random.randint(20)))
    utils.setElementProperty(key, str(inputids[1]), "maxVoltage", str(np.random.randint(20)))
    response = utils.startFor(key, 40)

    print("Fallin sleep", utils.time(key))
    steps = 20
    start = time.time()
    for i in range(steps):
        time.sleep(0.5)
        print(utils.time(key))

    time.sleep(10)
    print("Total time per step",time.time()-start,utils.time(key))
    utils.kill(key)

    return None

def main():
    N_parr=10

    p=Pool(N_parr)
    p.map(atomic_sim,list(range(40)))
    # with mp.pool.ThreadPool(processes=N_parr) as pool:
    #     outvals = pool.map(atomic_sim,2)



if __name__ == "__main__":
    main()