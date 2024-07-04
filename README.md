# Vibration Monitoring System

Este proyecto es un sistema de monitoreo de vibraciones diseñado para detectar fallas en maquinaria utilizando datos de sensores de vibración.

## Estructura del Proyecto

```plaintext
vibration-monitoring/
│
├── src/
│   ├── operation/
│   │   ├── __init__.py
│   │   ├── fault_detector.py
│   │   ├── gui.py
│   │   ├── spi_reader.py
│   │   └── main.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── gui.py
│   │   └── model_training.py
│   └── __init__.py
│
├── data/
│   ├── good_data.csv
│   ├── bad_data.csv
│   └── vibration_model.zip
│
├── README.md
└── requirements.txt
```
