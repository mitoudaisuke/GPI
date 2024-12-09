import serial.tools.list_ports

def get_usb_port_address():
    # 利用可能なシリアルポートのリストを取得
    ports = list(serial.tools.list_ports.comports())

    # シリアルポートの情報を表示
    for port in ports:
#        if port.description=='STM32 Virtual ComPort':
            print(port.device)

if __name__ == "__main__":
    get_usb_port_address()