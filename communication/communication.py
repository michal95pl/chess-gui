import socket
import json
from threading import Thread
from queue import Queue

class Communication(Thread):

    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        self.queue = Queue()
        self.start()

    def send(self, data: dict):
        self.socket.send(json.dumps(data).encode('utf-8'))

    def run(self):
        while True:
            data = self.socket.recv(4096)
            if data:
                message = json.loads(data.decode('utf-8'))
                self.queue.put(message)
                print(self.get_message())

    def get_message(self):
        if not self.queue.empty():
            return self.queue.get()
        return None

    def is_message_available(self):
        return not self.queue.empty()