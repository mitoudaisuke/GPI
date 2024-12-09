import serial
import time

def receive_callback(data):
    print(f"Received data: {data}")

def main():
    # VCPデバイスのポート名（Linux/Macでは "/dev/cu.usbmodem3495325C35381" など）
    port = "/dev/cu.usbmodem3495325C35381"

    try:
        # シリアルポートの設定
        ser = serial.Serial(port, baudrate=115200, timeout=1)

        while True:
            # PCからデータを入力
            data_to_send = input("Enter data to send: ")
            ser.write(data_to_send.encode())
            print(f"Sent data: {data_to_send}")

            # STM32からのエコーバックを受信
            data_received = ser.readline().decode().strip()
            print(f"Received data: {data_received}")

    except serial.SerialException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()