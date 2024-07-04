import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import model_training

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Trainer")
        
        self.good_data_file = ""
        self.bad_data_file = ""
        
        self.create_widgets()
        
    def create_widgets(self):
        tk.Label(self.root, text="Cargar Datos Buenos").grid(row=0, column=0, padx=10, pady=10)
        self.good_data_btn = tk.Button(self.root, text="Seleccionar Archivo", command=self.load_good_data)
        self.good_data_btn.grid(row=0, column=1, padx=10, pady=10)
        
        tk.Label(self.root, text="Cargar Datos Malos").grid(row=1, column=0, padx=10, pady=10)
        self.bad_data_btn = tk.Button(self.root, text="Seleccionar Archivo", command=self.load_bad_data)
        self.bad_data_btn.grid(row=1, column=1, padx=10, pady=10)
        
        self.train_btn = tk.Button(self.root, text="Entrenar Modelo", command=self.train_model)
        self.train_btn.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        self.export_btn = tk.Button(self.root, text="Exportar Modelo", command=self.export_model, state=tk.DISABLED)
        self.export_btn.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
        
        self.status_label = tk.Label(self.root, text="")
        self.status_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
        
    def load_good_data(self):
        self.good_data_file = filedialog.askopenfilename()
        if self.good_data_file:
            messagebox.showinfo("Información", "Datos buenos cargados correctamente.")
    
    def load_bad_data(self):
        self.bad_data_file = filedialog.askopenfilename()
        if self.bad_data_file:
            messagebox.showinfo("Información", "Datos malos cargados correctamente.")
    
    def train_model(self):
        if not self.good_data_file or not self.bad_data_file:
            messagebox.showerror("Error", "Debe cargar ambos archivos de datos buenos y malos.")
            return
        
        self.status_label.config(text="Entrenando el modelo, por favor espere...")
        self.good_data_btn.config(state=tk.DISABLED)
        self.bad_data_btn.config(state=tk.DISABLED)
        self.train_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=self.train_model_background).start()
    
    def train_model_background(self):
        try:
            X, y = model_training.load_data(self.good_data_file, self.bad_data_file)
            X_features, y_features = model_training.extract_features_and_labels(X, y)
            self.model = model_training.train_model(X_features, y_features)
            
            self.root.after(0, self.training_complete)
        except Exception as e:
            self.root.after(0, self.training_failed, str(e))
    
    def training_complete(self):
        messagebox.showinfo("Información", "Modelo entrenado correctamente.")
        self.status_label.config(text="Entrenamiento completado.")
        self.export_btn.config(state=tk.NORMAL)
        self.good_data_btn.config(state=tk.NORMAL)
        self.bad_data_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)
    
    def training_failed(self, error_message):
        messagebox.showerror("Error", error_message)
        self.status_label.config(text="Error en el entrenamiento.")
        self.good_data_btn.config(state=tk.NORMAL)
        self.bad_data_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.NORMAL)
    
    def export_model(self):
        export_file = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("Zip files", "*.zip")])
        if export_file:
            model_training.export_model(self.model, export_file)
            messagebox.showinfo("Información", "Modelo exportado y comprimido correctamente.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
