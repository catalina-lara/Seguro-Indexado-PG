# Seguro Indexado PG

Proyecto para el desarrollo de un seguro indexado asociado a variables climáticas y productivas del cultivo de café en Colombia.

**App desplegada:** [Página web](https://seguro-indexado-pg-iywupyyil9dkdmwmf3eq77.streamlit.app/)

- Vista Analista — KPIs, modelo, SHAP
- Vista Caficultor — más simple

## Datos

Los datos utilizados en este proyecto se encuentran almacenados en Google Drive debido a su tamaño.

### Data Raw

Contiene los datos originales utilizados en el proyecto.

### Data Clean

Contiene los datos procesados y preparados para el análisis.

📂 [Acceder a los datos](https://drive.google.com/drive/folders/18t1uUqwXNGSb2x4JbRFQ1Yj6Qsf7MLy1?usp=drive_link)

Antes de correr los notebooks, descarga esa carpeta y colócala en la raíz del proyecto, al mismo nivel que `scripts/` y `deploy/`, respetando la estructura `data_raw/` y `data_clean/`.

## Estructura del proyecto

\`\`\`
Seguro-Indexado-PG/
├── scripts/                          # Pipeline completo, en orden de ejecución
│   ├── Preprocesamiento.ipynb        # 1. Extracción y limpieza de fuentes crudas (GEE, Agronet, FNC)
│   ├── Preprocesamiento_2.ipynb      # 2. Ingeniería de variables, agregación anual-municipal
│   ├── Modelado.ipynb                # 3. Modelado de rendimiento (7 iteraciones), SHAP, exportación
│   └── Indice_parametrico.ipynb      # 4. Índice paramétrico, curva de payout, exportación para el dashboard
├── deploy/                           # App de Streamlit (generada por los notebooks 3 y 4)
│   ├── app.py
│   ├── requirements.txt
│   ├── .streamlit/config.toml
│   └── (8 artefactos generados automáticamente — ver abajo)
└── README.md
\`\`\`

## ¿Cómo reproducir los datos?

1. Descargar los datos de Google Drive (ver sección "Datos").
2. Instalar dependencias: `pip install -r deploy/requirements.txt`.
3. Correr los 4 notebooks de `scripts/` en el orden indicado arriba, de principio a fin (Restart & Run All en cada uno).
4. `Modelado.ipynb` e `Indice_parametrico.ipynb` generan automáticamente, dentro de `deploy/`, los 10 archivos que necesita la app: `config.json`, `metadata.json`, `diccionario_variables.json`, `metricas_modelo.json`, `modelo_rendimiento.pkl`, `historico_prediccion.json`, `features_modelo.json`, `promedios_municipio_features.json`.
5. Para correr la app localmente: `cd deploy && streamlit run app.py`

## Despliegue

La app está desplegada en Streamlit Community Cloud, conectada a la rama `main` de este repositorio — cualquier cambio subido a `deploy/` la redespliega automáticamente en 1-2 minutos.

## Documentación adicional

El manual de usuario, el anexo técnico y la rúbrica de evaluación diligenciada del proyecto están disponibles como documentos separados entregados junto con este repositorio.
