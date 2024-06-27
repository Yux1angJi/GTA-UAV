#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import zmq

class Client:
    def __init__(self, ip='127.0.0.1', port=8000, datasetPath=None, compressionLevel=0):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect("tcp://127.0.0.1:8000")

    def sendMessage(self, message):
        jsonstr = message.to_json()
        self.socket.send_string(jsonstr)
        return True

    def recvMessage(self):
        frame = self.socket.recv()
        data = self.socket.recv_string()
        message = json.loads(data)
        message['frame'] = frame
        return message

    def close(self):
        self.socket.close()










