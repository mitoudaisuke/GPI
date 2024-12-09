import serial.tools.list_ports
import threading
import tkinter as tk
from tkinter import ttk

class SerialMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Serial Monitor")

        self.selected_port = tk.StringVar()

        self.port_list_label = ttk.Label(root, text="利用可能なポート:")
        self.port_list_label.pack(pady=10)

        self.port_listbox = tk.Listbox(root, selectmode=tk.SINGLE)
        self.port_listbox.pack()

        self.refresh_button = ttk.Button(root, text="リフレッシュ", command=self.refresh_ports)
        self.refresh_button.pack(pady=10)

        self.select_button = ttk.Button(root, text="選択", command=self.select_port)
        self.select_button.pack(pady=10)

        self.serial_data_label = ttk.Label(root, text="シリアルデータ:")
        self.serial_data_label.pack(pady=10)

        self.serial_data_text = tk.Text(root, height=10, width=50, state=tk.DISABLED)
        self.serial_data_text.pack()

        self.thread = None

        self.refresh_ports()

    def refresh_ports(self):
        self.port_listbox.delete(0, tk.END)
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_listbox.insert(tk.END, port.device)

    def select_port(self):
        selected_index = self.port_listbox.curselection()
        if selected_index:
            selected_port = self.port_listbox.get(selected_index[0])
            self.selected_port.set(selected_port)
            self.start_serial_monitor()

    def start_serial_monitor(self):
        if self.thread and self.thread.is_alive():
            self.serial_data_text.insert(tk.END, "既にシリアルモニターが実行中です。\n")
            return

        selected_port = self.selected_port.get()
        if selected_port:
            self.thread = threading.Thread(target=self.read_serial_data, args=(selected_port,), daemon=True)
            self.thread.start()
        else:
            self.serial_data_text.insert(tk.END, "ポートが選択されていません。\n")

    def read_serial_data(self, selected_port):
        try:
            with serial.Serial(selected_port, 9600, timeout=1) as ser:
                self.serial_data_text.config(state=tk.NORMAL)
                self.serial_data_text.delete(1.0, tk.END)
                self.serial_data_text.insert(tk.END, f"{selected_port} でのシリアル通信を開始しました。\n")
                self.serial_data_text.config(state=tk.DISABLED)

                while True:
                    data = ser.readline().decode("utf-8").strip()
                    if data:
                        self.serial_data_text.config(state=tk.NORMAL)
                        self.serial_data_text.insert(tk.END, f"受信データ: {data}\n")
                        self.serial_data_text.yview(tk.END)
                        self.serial_data_text.config(state=tk.DISABLED)
        except serial.SerialException as e:
            self.serial_data_text.config(state=tk.NORMAL)
            self.serial_data_text.insert(tk.END, f"エラー: {e}\n")
            self.serial_data_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = SerialMonitorApp(root)
    root.mainloop()
