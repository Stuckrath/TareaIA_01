import tkinter as tk
from GUI import SimuladorGUI

if __name__ == "__main__":
    # Inicializa la ventana principal de Tkinter
    root = tk.Tk()
    
    # Instancia la aplicación gráfica
    app = SimuladorGUI(root)
    
    # Inicia el bucle principal de eventos
    root.mainloop()