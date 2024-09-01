# Autosimilitud en redes

El repositorio se encuentra organizado de la siguiente forma:

* ```Self similarity in graphs.ipynb```: notebook principal. Contiene todo el flujo de ejecuciones necesario para llegar a los resultados presentados en la memoria del Trabajo de Fin de Máster. También se presenta en formato '.pdf'.
* ```utils_graph.py```: funciones auxiliares del notebook de creación de grafo, obtención de métricas...
* ```nestedness_calculator.py```: implementa el algoritmo NODF para el calculo del coeficiente de anidamiento. Creado por  Mika Straka.
* ```requirements.txt```: archivo con las librerías empleadas para la ejecución de los archivos de código.
* ```d-mercator/```: directorio que contiene todos los archivos necesarios para ejecutar la herramienta de d-mercator en las redes.
        
        ├── d-mercator
        │    ├── graphs: grafos con nodos como hashtags filtrados de las diferentes manifestaciones analizadas
        |    │   ├── MANIFESTACION: grafos de la manifestacion MANIFESTACION
        |    │   |   ├── HORA: carpeta de los grafos derivados de la hora HORA
        |    │   |   |   ├── UMBRAL: archivos del grafo creado con un umbral de peso UMBRAL. El archivo HORA.edge contiene la información de las aristas del grafo. El resto de archivos son produdcidos por d-mercator al ejecutar el script d-mercator.py.
        │    ├── d-mercator.py: lanza la herramienta de d-mercator y genera los archivos de estadísticas en la carpeta correspondiente
        │    ├── create_graph_file_dmercator.ipynb: notebook para generar los archivos de grafo de forma que d-mercator sea capaz de leerlos adecuadamente.

        |    │   ├── nat: grafos de No al Tarifazo
* ```datasets/```: directorio que contiene los conjuntos de datos obtenidos de Twitter de ambos movimientos sociales, No al Tarifazo (o nat) y 9 de noviembre (o 9n)
* ```measures/```: directorio que contiene las diferentes métricas calculadas para cada grafo, para evitar tener que recalcularlas. Está estructurado de la siguiente forma:
        
        ├── measures: archivos con información general de anidamiento y modularidad para cada manifestación y estrategia de formación de redes (hashtag, usuarios y bipartitas)
        │    ├── 9n: archivos, por horas, con información del coeficiente medio de clusterización por K_t (avg_clust), coeficiente de clusterización medio de nodos con el mismo internal degree para cada K_t (int_deg) y distribución de grados de la manifestación 9 de noviembre (des_kt) para cada hora y cada estrategia de creación de grafos  (h, hashtags como nodos, u, usuarios como nodos, b, redes bipartitas o f, grafo original filtrando por umbral de peso).
        │    ├── nat: archivos, por horas, con información del coeficiente medio de clusterización por K_t, coeficiente de clusterización medio de nodos con el mismo internal degree para cada K_t y distribución de grados de la manifestación No al Tarifazo.
        |    ├── MANIFESTACION_MODE.json: archivo de la MANIFESTACION (nat o 9n) para cada estrategia de generación (h, hashtags como nodos, u, usuarios como nodos, b, redes bipartitas)
* ```plots/```: contiene los diferentes gráficos generados en la ejecución del notebook principal.
* ```graphs/```: directorio con los diferentes grafos generados en formato ```.gexf```. Está estructurado de la siguiente forma:
        
        ├── graphs
        │    ├── bipartite: grafos bipartitos 
        |    │   ├── 9n: grafos de 9 de noviembre
        |    │   ├── nat: grafos de No al Tarifazo
        │    ├── filtered: grafos con el filtro por umbral de peso de arista
        |    |   ├── UMBRAL
        |    │   │   ├── MANIFESTACION: grafos filtrados con umbral de peso UMBRAL para la manifestación MANIFESTACIÓN para las diferentes horas.
        │    ├── nodes_hashtag: grafos con hashtags como nodos
        |    │   ├── 9n: grafos de 9 de noviembre
        |    │   ├── nat: grafos de No al Tarifazo
        │    ├── nodes_user: grafos con usuarios como nodos
        |    │   ├── 9n: grafos de 9 de noviembre
        |    │   ├── nat: grafos de No al Tarifazo
    