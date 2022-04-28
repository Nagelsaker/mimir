import socket
import time
import numpy as np

def main():
    UDP_IP = "10.42.0.1"
    UDP_PORT = 20000


    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) # UDP
    sock.bind((UDP_IP, UDP_PORT))


    while True:
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        angle = np.rad2deg(float(data))
        print(f"Measured angle: {angle:.2f}\t Data: {data}")

if __name__ == "__main__":
    main()