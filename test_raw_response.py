#!/usr/bin/env python3
import requests
import json
import time

time.sleep(2)

print('Full API Response Test')
with open('sample_plate.jpg', 'rb') as f:
    resp = requests.post('http://localhost:5000/predict', files={'image': f}, timeout=30)

data = resp.json()
print(json.dumps(data, indent=2))
