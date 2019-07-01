import serial
import argparse

class Blink:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.arduinoSerialData = serial.Serial()
        self.arduinoSerialData.port = port
        self.arduinoSerialData.baudrate = baudrate
        self.arduinoSerialData.timeout = 1
        self.arduinoSerialData.setDTR(False)
        #arduinoSerialData.setRTS(False)

    def openConnection(self):
        self.arduinoSerialData.open()

    def closeConnection(self):
        self.arduinoSerialData.close()

    def blinkLED(self, ledPin='3'):
        self.arduinoSerialData.write(str.encode(ledPin))

if __name__== "__main__":
    ''' use main for testing '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", required=True, help="need com port: windows ex. COM3: Linux ex. /dev/ttyUSB0")
    args = vars(ap.parse_args())
    port = args["port"]
    b = Blink(port=port)
    b.openConnection()
    b.blinkLED()
    b.closeConnection()
