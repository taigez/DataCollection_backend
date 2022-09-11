import requests  
import json

url = "http://10.251.8.66:8000/classifier/receive_json/"
data = {"text":"Marc Snir received a Ph.D. in Mathematics from the Hebrew University of Jerusalem in 1979, worked at NYU on the NYU Ultracomputer project in 1980-1982, and was at the Hebrew University of Jerusalem in 1982-1986, before joining IBM. Marc Snir was a major contributor to the design of the Message Passing Interface. He has published numerous papers and given many presentations on computational complexity, parallel algorithms, parallel architectures, interconnection networks, parallel languages and libraries and parallel programming environments."}
headers = {'content-type': 'application/json'}
r=requests.post(url, data=json.dumps(data), headers=headers)
r.text