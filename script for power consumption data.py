import threading
import serial
import time



class SerialReader(threading.Thread):
    def __init__(self, port, name, baudrate=230400):
        super().__init__()

        self.serial_port = serial.Serial(port, baudrate)
        self.name = name 

    def run(self):
        while 1:
            response = self.serial_port.readline()
            if len(response) == 11:
                #I_IN = self.get_val(response[0], response[1])
                #V_IN = self.get_val(response[2], response[3])
                I_OUT = self.get_val(response[4], response[5])
                V_OUT = self.get_val(response[6], response[7])
                #TEMP = self.get_val(response[8], response[9])

                with open(self.name, "a") as t:
                    t.write(f"{time.time()}, {I_OUT * V_OUT:.4f}\n")

#{time.time()} excluded from t.write

    def stop(self):
        self.running = False
        self.serial_port.close()


    def get_val(self, high, low):
        val = (high << 8) + low
        return (val / 100)
 



serial_reader = SerialReader(port = "COM14", name = "Idle.txt")

serial_reader.start()
