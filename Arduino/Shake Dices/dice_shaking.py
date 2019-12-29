from time import sleep
import serial


def shake_once():
    arduino = serial.Serial('COM5', 9600, timeout=.1)
    arduino.close()
    arduino.open()
    sleep(3)
    print(arduino.name)
    print(arduino.readline())
    arduino.write(b't')
    sleep(3)
    print(arduino.readlines())
    arduino.write(b'f')
    arduino.close()


# shake_once()
