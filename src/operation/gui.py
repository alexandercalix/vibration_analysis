import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from fault_detector import FaultDetector
from spi_reader import SPIReader
import threading
import time
import os
import numpy as np

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vibration Monitoring System")
        
        self.log_table = ttk.Treeview(self, columns=("Time", "Message"), show="headings")
        self.log_table.heading("Time", text="Time")
        self.log_table.heading("Message", text="Message")
        self.log_table.pack(fill=tk.BOTH, expand=True)
        
        self.state_label = tk.Label(self, text="State: All Good", bg="green", fg="white")
        self.state_label.pack(fill=tk.X)
        
        self.sample_count_label = tk.Label(self, text="Samples: 0")
        self.sample_count_label.pack(fill=tk.X)
        
        self.monitoring = False
        self.data = []
        self.sample_count = 0
        self.test_interval = 100
        self.update_interval = 0.1
        self.spi_reader = None
        self.fault_detector = None
        self.trend_window = 10  # Ventana para calcular la tendencia
        self.threshold = 0.5  # Umbral de cambio significativo
        
        self.start_button = tk.Button(self, text="Start Monitoring", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.stop_button = tk.Button(self, text="Stop Monitoring", command=self.stop_monitoring)
        self.stop_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.load_model_button = tk.Button(self, text="Load Model", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT, padx=10, pady=10)

    def start_monitoring(self):
        if not self.fault_detector:
            messagebox.showwarning("Warning", "Load a model before starting monitoring.")
            return
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor)
        self.monitor_thread.start()
        print("Monitoring started")

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("Monitoring stopped")
    
    def load_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("Model files", "*.zip")])
        if model_path:
            self.fault_detector = FaultDetector(model_path)
            self.log_message("Model loaded successfully.")
            print("Model loaded successfully.")
    
    def update_sample_count_label(self):
        self.sample_count_label.config(text=f"Samples: {self.sample_count}")
        print(f"Sample count updated: {self.sample_count}")

    def handle_monitoring_error(self, error):
        self.after(0, messagebox.showerror, "Error", f"Error during monitoring: {error}")
        self.monitoring = False
        print(f"Error during monitoring: {error}")

    def log_message(self, message):
        self.log_table.insert("", 0, values=(datetime.now(), message))
        print(f"Log message: {message}")

    def update_interface(self, prediction):
        print("Updating interface with prediction")
        try:
            fault_types = {
                1: "Bearing",
                2: "Motor Shaft",
                3: "Transmission",
                4: "Unbalance",
                5: "Misalignment"
            }
            
            if prediction != 0:
                print("Fault detected, prediction:", prediction)
                fault_type = fault_types.get(prediction, "Unknown")
                self.log_table.insert("", 0, values=(datetime.now(), f"Fault detected: {fault_type}"))
                self.state_label.config(text=f"State: Fault Detected - {fault_type}", bg="red", fg="white")
            else:
                print("No fault detected")
                self.log_table.insert("", 0, values=(datetime.now(), "No fault detected"))
                self.state_label.config(text="State: All Good", bg="green", fg="white")
        except Exception as e:
            self.log_message(f"Error updating interface: {e}")
            print(f"Error updating interface: {e}")

    def monitor(self):
        try:
            self.spi_reader = SPIReader(fault_probability=0.1)
            recent_features = []
            while self.monitoring:
                data = self.spi_reader.read_vibration_data()
                print(f"Simulated data: {data}")
                if data:
                    self.data.append(data)
                    self.sample_count += 1
                    self.after(0, self.update_sample_count_label)
                    if len(self.data) >= self.test_interval:
                        try:
                            features = self.fault_detector.extract_features(self.data)
                            print(f"Extracted features: {features}")
                            prediction = self.fault_detector.detect_fault(features)
                            print(f"Prediction: {prediction}")
                            self.after(0, self.update_interface, prediction[0])
                            
                            recent_features.append(features.mean().values)
                            if len(recent_features) > self.trend_window:
                                recent_features.pop(0)
                            
                            if len(recent_features) == self.trend_window:
                                trends = np.mean(recent_features, axis=0)
                                if any(abs(trends) > self.threshold):
                                    self.log_table.insert("", 0, values=(datetime.now(), f"Predictive alarm: Significant trend change detected"))
                                    print("Predictive alarm: Significant trend change detected")
                            
                        except ValueError as e:
                            self.after(0, self.log_message, str(e))
                            print(f"ValueError: {e}")
                        self.data = []
                        self.sample_count = 0
                time.sleep(self.update_interval)
        except Exception as e:
            self.after(0, self.handle_monitoring_error, e)
            print(f"Exception in monitor: {e}")
        finally:
            if self.spi_reader:
                self.spi_reader.close()
                print("SPI Reader closed")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
