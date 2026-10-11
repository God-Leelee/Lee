"""Local delivery verification with cached weights and Python outbound guard."""
import os
import socket
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
attempts=[]
original_connect=socket.socket.connect
original_connect_ex=socket.socket.connect_ex


def allow(address):
    if not isinstance(address,tuple):return True
    if address[0] in {'127.0.0.1','::1','localhost'}:return True
    attempts.append({'host':str(address[0]),'port':address[1]})
    (ROOT/'runs/offline_network_attempts.json').write_text(json.dumps(attempts,ensure_ascii=False,indent=2),encoding='utf-8')
    return False


def guarded_connect(self,address):
    if not allow(address):raise OSError('Outbound TCP denied during local offline verification')
    return original_connect(self,address)


def guarded_connect_ex(self,address):
    if not allow(address):return 10013
    return original_connect_ex(self,address)


socket.socket.connect=guarded_connect
socket.socket.connect_ex=guarded_connect_ex
(ROOT/'runs').mkdir(exist_ok=True)
(ROOT/'runs/offline_network_attempts.json').write_text('[]',encoding='utf-8')
if __name__=='__main__':
    import uvicorn
    uvicorn.run('backend.app:app',host='127.0.0.1',port=8765,access_log=False)
