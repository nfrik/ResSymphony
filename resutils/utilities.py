from __future__ import print_function

import requests
import json
import time
import gzip
import base64

class Utilities:

    serverUrl = ""

    def __init__(self, serverUrl):
        Utilities.serverUrl=serverUrl


    # Create new simulation and retrieve unique simulation key
    def createNewSimulation(self):
        payload = ""
        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        url = Utilities.serverUrl+"simulations"

        payload = ""

        response = requests.request("POST",url,data=payload,headers=headers)

        return json.loads(json.dumps(response.text))

    # Get all currently running simulations
    def getSimulations(self):
        url = Utilities.serverUrl + "simulations"

        response = requests.request("GET", url)

        return json.loads(json.dumps(response.text))

    # Append graph elements to current circuit
    def addGraphElements(self,key,elementsJsonString):
        files = {'file': ('circuit.json', elementsJsonString)}

        url = Utilities.serverUrl + "simulations/" + key + "/addGraphElementsFromFile"

        response = requests.request("POST", url, files=files)

        return json.loads(json.dumps(response.text))


    # Append graph elements to current circuit from file
    def addGraphElementsFromFile(self,key,jsonFile):
        files = {'file': open(jsonFile, 'rb')}

        url = Utilities.serverUrl + "simulations/" + key + "/addGraphElementsFromFile"

        response = requests.request("POST", url, files=files)

        return json.loads(json.dumps(response.text))

    # Returns currently loaded graph circuit as JSON string
    def currentCircuitAsJSONGraph(self,key):
        #post
        url = Utilities.serverUrl + "simulations/" + key + "/currentCircuitAsJSONGraph"

        response = requests.request("GET", url)

        return json.loads(json.dumps(response.text))



    # Delete graph elements of a circuit
    def deleteGraphElement(self,key,elementId):
        url = Utilities.serverUrl + "simulations/" + key + "/startForAndWait"
        querystring = {"elementId": elementId}

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers, params=querystring)

        return json.loads(json.dumps(response.text))

    # List all available element properties
    def getElementProperties(self,key,elementId):
        url = Utilities.serverUrl + "simulations/" + key + "/element/"+elementId
        querystring = {"elementId": elementId}

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers, params=querystring)

        return json.loads(json.dumps(response.text))

    # Get element property
    def getElementProperty(self, key, elementId,propertyKey):
        url = Utilities.serverUrl + "simulations/" + key + "/element/" + elementId+"/property"
        querystring = {"propertyKey": propertyKey}

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers, params=querystring)

        return json.loads(json.dumps(response.text))

    def getElementsIVs(self, key):
        # post
        url = Utilities.serverUrl + "simulations/" + key + "/elements"

        response = requests.request("GET", url)

        return json.loads(json.dumps(response.text))

    # Changes specified element property to a given value
    def setElementProperty(self,key,elementId, propertyKey, newValue):
        url = Utilities.serverUrl + "simulations/" + key + "/element/" + elementId+"/property"
        querystring = {"propertyKey": propertyKey,"newValue":newValue}

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.patch(url, headers=headers, params=querystring)

        return json.loads(json.dumps(response.text))

    # POST / symphony / simulations / {key} / element / {elementId} / current
    #  Returns current for specified element (real time)
    def getCurrent(self,key,elementId):
        url = Utilities.serverUrl + "simulations/" + key + "/element/" + elementId + "/current"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers)

        return json.loads(json.dumps(response.text))

    # POST / symphony / simulations / {key} / element / {elementId} / peekCurrent
    # Returns current on specified element when last measurement was performed
    def peekCurrent(self, key, elementId):
        url = Utilities.serverUrl + "simulations/" + key + "/element/" + elementId + "/peekCurrent"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers)

        return json.loads(json.dumps(response.text))

    # Returns voltage diff on specified element when last measurement was performed
    def peekVoltageDiff(self,key,elementId):
        url = Utilities.serverUrl + "simulations/" + key + "/element/" + elementId + "/peekVoltageDiff"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers)

        return json.loads(json.dumps(response.text))

    # Returns voltage diff on specified element (real time)
    def getVoltageDiff(self,key,elementId):
        url = Utilities.serverUrl + "simulations/" + key + "/element/" + elementId + "/voltageDiff"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers)

        return json.loads(json.dumps(response.text))

    # Returns currently loaded graph circuit as CMF
    def getGraphAsCmf(self,key):
        return None

    # Load circuit from uploaded CMF file
    def loadCircuit(self,key,cmfFilePath):
        files = {'file':open(cmfFilePath,'rb')}
        #files = {'file': ('report.csv', 'some,data,to,send\nanother,row,to,send\n')}

        url = Utilities.serverUrl + "simulations/" + key + "/loadCircuit"

        response = requests.request("POST", url, files=files)

        return json.loads(json.dumps(response.text))

    # Load circuit represented by graph from uploaded JSON file
    def loadCircuitFromGraphFile(self,key,jsonFile):
        files = {'file': open(jsonFile, 'rb')}

        url = Utilities.serverUrl + "simulations/" + key + "/loadCircuitFromGraph"

        response = requests.request("POST", url, files=files)

        return json.loads(json.dumps(response.text))

    # Load circuit represented by graph from uploaded JSON file
    def loadCircuitFromGraphString(self, key, jsonString):
        files = {'file': ('circuit.json', jsonString)}

        url = Utilities.serverUrl + "simulations/" + key + "/loadCircuitFromGraph"

        response = requests.request("POST", url, files=files)

        return json.loads(json.dumps(response.text))

    # Retrieve measurements from specified simulation as response file

    def measurements(self,key):
        # post
        url = Utilities.serverUrl + "simulations/" + key + "/measurements_json"

        response = requests.request("GET", url)

        return json.loads(json.dumps(response.text))

    def measurements_gzip(self,key):
        # post
        url = Utilities.serverUrl + "simulations/" + key + "/measurements_gzip"

        response = requests.request("GET", url)

        json_response = json.loads(json.loads(json.dumps(response.text)))

        b64_response = base64.b64decode(json_response['message'])

        str_response = gzip.decompress(b64_response).decode("utf-8")

        return json.dumps({'key': key, 'measurements': json.loads(str_response)})

    # Changes peek interval of measurements
    def settings(self,key, peekInterval, pokeInterval):

        url = Utilities.serverUrl + "simulations/" + key + "/settings"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.request("PATCH", url, headers=headers,
                                    params={'peekInterval': peekInterval, 'pokeInterval': pokeInterval})

        return json.loads(json.dumps(response.text))

    def setArbWaveData(self, key, jsonString):
        # files = {'file': ('circuit.json', jsonString)}
        url = Utilities.serverUrl + "simulations/" + key + "/setArbwaveData"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.request("POST", url=url, headers=headers, data=jsonString)

        return json.loads(json.dumps(response.text))

    def setMeasurableElements(self, key, jsonString):
        # files = {'file': ('circuit.json', jsonString)}
        url = Utilities.serverUrl + "simulations/" + key + "/setMeasurableElements"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.request("POST", url=url, headers=headers, data=jsonString)

        return json.loads(json.dumps(response.text))

    # Returns simulation time when last measurement was performed
    def peekTime(self,key):
        url = Utilities.serverUrl + "simulations/" + key + "/peekTime"
        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.request("POST", url, headers=headers)

        return json.loads(json.dumps(response.text))

    # Returns simulation time when last measurement was performed
    def time(self,key):
        url = Utilities.serverUrl + "simulations/" + key + "/time"
        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.request("POST", url, headers=headers)

        return json.loads(json.dumps(response.text))

    # Starts simulation and returns immediately
    def start(self,key):
        url = Utilities.serverUrl + "simulations/" + key + "/start"
        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.request("POST", url, headers=headers)

        return json.loads(json.dumps(response.text))

    # Starts simulation for specified period of simulated circuit seconds and returns immediately
    def startFor(self,key,seconds):
        url = Utilities.serverUrl + "simulations/" + key + "/start"
        querystring = {"seconds":seconds}

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers,params=querystring)

        return json.loads(json.dumps(response.text))

    # Starts simulation for specified period of simulated circuit seconds and waits for it to finish
    # before completing the request.
    def startForAndWait(self,key,seconds):
        url = Utilities.serverUrl + "simulations/" + key + "/startForAndWait"
        querystring = {"seconds": seconds}

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers, params=querystring)

        return json.loads(json.dumps(response.text))

    # Stops simulation, after this it can be still resumed
    def stop(self,key):
        url = Utilities.serverUrl + "simulations/" + key + "/stop"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers)

        return json.loads(json.dumps(response.text))

    # Kills simulation, after this it will become inaccessible
    def kill(self,key):
        url = Utilities.serverUrl + "simulations/" + key + "/kill"

        headers = {
            'content-type': "application/json",
            'accept': "application/json"
        }

        response = requests.post(url, headers=headers)

        return json.loads(json.dumps(response.text))

def main():
    general_test()

    # voltage_set_test()

    # current_dynamic_test()

def voltage_set_test():
    utils = Utilities(serverUrl="http://localhost:8090/symphony/")


    response = utils.createNewSimulation()
    print(response)
    key = json.loads(response)["key"]


    jsonstr = json.dumps(json.load(open("/home/nifrick/PycharmProjects/ressymphony/resources/voltage_set_test.json")))
    response = utils.loadCircuitFromGraphString(key, jsonstr)
    print(response)

    utils.start(key)

    response = utils.getElementProperty(key, "1", "maxVoltage")
    print("Initial maxVoltage for source:"+response)

    response = utils.getVoltageDiff(key, "2")
    print("Voltage should be 0.0: " + response)

    response = utils.setElementProperty(key, "1", "maxVoltage", "-100")
    print(response)

    response = utils.getElementProperty(key, "1", "maxVoltage")
    print("After set maxVoltage for source should be 100.5:" + response)

    response = utils.getCurrent(key, "2")
    print("Voltage should be 100.0: " + response)

def current_dynamic_test():
    utils = Utilities(serverUrl="http://localhost:8090/symphony/")

    response = utils.createNewSimulation()
    print(response)
    key = json.loads(response)["key"]

    jsonstr = json.dumps(json.load(open("/home/nifrick/PycharmProjects/ressymphony/resources/current_dynamics_test.json")))
    response = utils.loadCircuitFromGraphString(key, jsonstr)
    print(response)

    utils.start(key)

    for i in range(10):
        response = utils.getCurrent(key, "1");
        print("Current should be different: " + response);
        time.sleep(0.1)



def general_test():
    utils = Utilities(serverUrl="http://localhost:8090/symphony/");
    response = utils.createNewSimulation()
    print(response)
    key = json.loads(response)["key"]

    response = utils.loadCircuit(key,"/home/nifrick/IdeaProjects/CircuitSymphony/src/test/resources/transistor_deleteme.json.cmf")
    print(response)

    response = utils.loadCircuitFromGraphFile(key,"/home/nifrick/IdeaProjects/CircuitSymphony/src/test/resources/transistor_a.json")
    print(response)

    jsoncontent = json.dumps(json.load(open("/home/nifrick/IdeaProjects/CircuitSymphony/src/test/resources/transistor_a.json")))
    response = utils.loadCircuitFromGraphString(key,jsoncontent)
    print(response)

    response = utils.start(key)
    print(response)

    response = utils.startFor(key, 1)
    print(response)

    realt = float(json.loads(utils.time(key))["time"])
    response = utils.startForAndWait(key, 10)
    realt = float(json.loads(utils.time(key))["time"]) - realt
    print("Waiting to equilibrate: SIM {} secs".format(realt))
    print(response)

    jsoncontens = json.dumps(json.load(open("/home/nifrick/IdeaProjects/CircuitSymphony/src/test/resources/transistor_b.json")))
    response = utils.addGraphElements(key,jsoncontens)
    print(response)

    response = utils.addGraphElementsFromFile(key,"/home/nifrick/IdeaProjects/CircuitSymphony/src/test/resources/transistor_b.json")
    print(response)

    response = utils.currentCircuitAsJSONGraph(key)
    print(response)

    response = utils.getElementProperties(key,"1")
    print(response)

    response = utils.getElementProperty(key, "1", "beta")
    print(response)

    response = utils.setElementProperty(key, "1","beta","101")
    print(response)

    response = utils.getElementProperty(key, "1", "beta")
    print(response)

    response = utils.getCurrent(key, "1")
    print(response)

    response = utils.peekCurrent(key, "1")
    print(response)

    response = utils.getVoltageDiff(key, "1")
    print(response)

    response = utils.peekVoltageDiff(key, "1")
    print(response)


if __name__ == "__main__":
    main()