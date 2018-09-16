from resutils import utilities
import json

utils = utilities.Utilities(serverUrl="http://localhost:8090/symphony/")

lst =json.loads(utils.getSimulations())

for item in lst:
    print("Killing key: ",item['key'])
    utils.kill(item['key'])
