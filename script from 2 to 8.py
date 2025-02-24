import serial
import time
import struct 

ser = serial.Serial(port="COM7", baudrate= 115200)
time.sleep(3)

for i in range(1):

    sera = [150, 78, 68, 170, 75, 75]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(2)

    sera = [150, 58, 68, 170, 75, 75]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(2)

    sera = [150, 58, 120, 170, 75, 75]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(2)

    sera = [150, 58, 120, 85, 75, 75]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(2)

    sera = [150, 58, 120, 85, 75, 35]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(2)

    sera = [150, 58, 120, 170, 75, 35]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(2)

    sera = [150, 58, 68, 170, 75, 35]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(2)

    sera = [150, 78, 68, 170, 75, 35]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(2)

    sera = [30, 78, 68, 170, 75, 35]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(1)

    sera = [30, 58, 68, 170, 75, 35]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(1)

    sera = [30, 58, 120, 170, 75, 35]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(1)

    sera = [30, 58, 120, 95, 75, 35]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(1)

    sera = [30, 58, 120, 95, 75, 75]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))
    time.sleep(1)

    sera = [90, 78, 68, 170, 75, 75]
    val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
    ser.write(bytes(val1, encoding ="utf-8"))

# sera = [90,90,90,90,90,90]
# val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
# ser.write(bytes(val1, encoding ="utf-8"))
# time.sleep(1)

# sera = [180,180,180,180,180,180]
# val1 = f"{sera[0]}-{sera[1]}-{sera[2]}-{sera[3]}-{sera[4]}-{sera[5]}/"
# ser.write(bytes(val1, encoding ="utf-8"))
# time.sleep(1)
