""" Archivo auxiliar con funciones necesarias en la ejecución del notebook principal"""

import networkx as nx
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
import powerlaw


from nestedness_calculator import NestednessCalculator


########################################################################
#
# CREACION DE GRAFOS
#
########################################################################

def read_data(manifestacion, datasets_folder = "datasets/"):
    """
    Lee un archivo de datos en formato CSV delimitado por espacios y lo carga en un DataFrame de Pandas.

    Parámetros:
    -----------
    manifestacion : str
        Nombre del archivo de datos (sin la extensión) que se encuentra en la carpeta especificada.
    
    datasets_folder : str, opcional, por defecto 'datasets/'
        Ruta relativa de la carpeta que contiene los archivos de datos. 
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame de Pandas que contiene los datos leídos desde el archivo CSV.

    Ejemplo:
    --------
    df = read_data("datos_ejemplo")
    """
    return pd.read_csv(datasets_folder + manifestacion + ".txt", sep= ' ')

def create_bipartite_graph(df, manifestacion, graphs_folder="graphs/"):
    """
    Crea y guarda grafos bipartitos basados en las relaciones entre usuarios y hashtags, 
    segmentados por hora, a partir de un DataFrame dado.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene la información de usuarios, hashtags y horas. 
        Se espera que las columnas incluyan al menos 'user', 'hashtag', 'hour' y 'weight'.
    
    manifestacion : str
        Nombre de la manifestación o evento, utilizado para organizar los archivos generados.
    
    graphs_folder : str, opcional, por defecto 'graphs/'
        Ruta relativa de la carpeta donde se almacenarán los grafos generados. Los grafos se guardan en
        una subcarpeta 'bipartite/{manifestacion}/' dentro de esta carpeta.

    Retorna:
    --------
    None
        La función no retorna ningún valor. Los grafos bipartitos son guardados como archivos .gexf 
        en la ruta especificada.

    Ejemplo:
    --------
    create_bipartite_graph(df, "protesta_2024")
    """
    graphs_folder = graphs_folder + 'bipartite/' + manifestacion + '/'
    df_h = df["hour"].unique()
    print("Creando redes bipartitas, manifestación seleccionada:", manifestacion, "número de horas: ", len(df_h))
    G = nx.Graph()
    for hour in df_h:
        df_hour = df[(df["hour"] == hour)]
        G = nx.from_pandas_edgelist(df_hour, source="user", target="hashtag", edge_attr="weight")
        nx.write_gexf(G, graphs_folder + str(hour) + ".gexf")

def create_graphs(node_criteria, edge_criteria, df, manifestacion, graphs_folder="graphs/"):
    """
    Genera y almacena grafos basados en las relaciones entre nodos, que pueden ser usuarios o hashtags, según 
    el criterio especificado. Los grafos se crean para cada hora distinta en el DataFrame, y pueden incluir una 
    opción para filtrar aristas con pesos inferiores a un umbral dado.

    Parámetros:
    -----------
    node_criteria : str
        Criterio para los nodos de la red, puede ser 'user' o 'hashtag'. Determina qué tipo de entidad (usuarios 
        o hashtags) se representará como nodos en el grafo.
    
    edge_criteria : str
        Criterio para las aristas de la red, puede ser 'user' o 'hashtag'. Define cómo se conectarán los nodos 
        en base a entidades compartidas (hashtags o usuarios).

    df : pd.DataFrame
        DataFrame que contiene la información de usuarios, hashtags, horas y pesos de las conexiones. 
        Debe incluir las columnas correspondientes a 'user', 'hashtag', 'hour', y 'weight'.
    
    manifestacion : str
        Identificador de la manifestación o evento, utilizado para estructurar las carpetas y nombres de los archivos generados.
    
    graphs_folder : str, opcional, por defecto 'graphs/'
        Directorio base donde se guardarán los grafos generados. Se creará una subcarpeta en 'nodes_{node_criteria}/{manifestacion}/'
        para almacenar los archivos correspondientes a cada hora.

    Retorna:
    --------
    None
        Esta función no retorna un valor. Los grafos generados se almacenan como archivos .gexf en la estructura de carpetas especificada.

    Ejemplo de uso:
    --------------
    create_graphs("user", "hashtag", df, "protesta_2024")
    """
    graphs_folder = graphs_folder + 'nodes_' + node_criteria + '/'+ manifestacion + '/'
    df_h = df["hour"].unique()
    print("Creando redes de", node_criteria, "unidos si comparten uno o más", edge_criteria, ", manifestación seleccionada:", manifestacion, "número de horas: ", len(df_h))
    for hour in df_h:
        G = nx.Graph()
        df_hour = df[(df["hour"] == hour)]
        df_nodes = df_hour[node_criteria].unique()
        G.add_nodes_from(df_nodes)
        for node in df_nodes:
            # Seleccionamos las filas del dataframe con el usuario/hashtag sobre el que iteramos
            df_node_edge = df_hour.loc[df_hour[node_criteria] == node]
            # Seleccionamos tantos hashtags/usuarios como haya que haya compartido el usuario/hasthag respectivamente
            df_node_edge = df_node_edge[edge_criteria]
            for edge in df_node_edge:
                df_edge = df_hour.loc[df_hour[edge_criteria] == edge]
                df_edge = df_edge[node_criteria]
                for nd in df_edge:
                    if nd != node:
                        if G.has_edge(node, nd):
                            G[node][nd]["weight"] += 1
                        else:
                            G.add_edge(node, nd, weight = 1)
        
        # Finalmente dividimos entre dos todos los pesos de las aristas, pues están contados dos veces (uno por cada nodo)
        for edge in G.edges():
            old_weight = G.edges[edge]["weight"]
            nx.set_edge_attributes(G, {edge: {"weight": old_weight/2}})

        nx.write_gexf(G, graphs_folder + str(hour) + ".gexf")

def calc_avg_degree(G):
    """
    Calcula el grado medio (average degree) de un grafo no dirigido.

    Parámetros:
    -----------
    G : networkx.Graph
        Grafo no dirigido del cual se calculará el grado medio. Los nodos pueden representar cualquier entidad,
        y las aristas pueden tener cualquier tipo de relación.

    Retorna:
    --------
    float
        El grado medio del grafo, calculado como la suma de los grados de todos los nodos dividida 
        entre el número total de nodos en el grafo.

    Ejemplo de uso:
    --------------
    avg_degree = calc_avg_degree(G)
    """
    return sum(dict(G.degree).values())/G.number_of_nodes()

def add_nodes_subgraph(G, threshold):
    """
    Crea un subgrafo que incluye solo los nodos cuyo grado excede un umbral especificado.

    Parámetros:
    -----------
    G : networkx.Graph
        Grafo original del cual se extraerán los nodos para el subgrafo. 
        Los nodos representan entidades y las aristas relaciones entre ellas.

    threshold : int
        Umbral de grado mínimo que un nodo debe tener para ser incluido en el subgrafo.
    
    Retorna:
    --------
    networkx.Graph
        Un nuevo subgrafo que contiene únicamente los nodos del grafo original cuyo grado 
        es mayor que el umbral especificado. Las aristas originales no se conservan en este subgrafo.

    Ejemplo de uso:
    --------------
    subgraph = add_nodes_subgraph(G, threshold=10)
    """
    F = nx.Graph()
    for node in G.nodes():
        # Se comprueba si el grado del nodo es mayor que el umbral y se añade
        if G.degree[node] > threshold:
            F.add_node(node)
    return F

def add_edges_subgraph(G, F):
    """
    Añade aristas a un subgrafo `F` a partir de un grafo original `G`. Si dos nodos en `F` están conectados 
    en `G`, se añade la correspondiente arista a `F`.

    Parámetros:
    -----------
    G : networkx.Graph
        Grafo original que contiene tanto los nodos como las aristas entre ellos.
    
    F : networkx.Graph
        Subgrafo que contiene un subconjunto de nodos de `G`. Esta función añadirá a `F` 
        las aristas que conectan esos nodos en `G`.

    Retorna:
    --------
    networkx.Graph
        El subgrafo `F` actualizado, que contiene las aristas correspondientes a las conexiones 
        entre los nodos de `F` en el grafo original `G`.

    Ejemplo de uso:
    --------------
    F = add_edges_subgraph(G, F)
    """
    for node in F.nodes():
        # Se itera sobre los vecinos en G de cada nodo y se ve si pertenecen a F
        for neighbor in G.neighbors(node):
            if neighbor in F.nodes():
                # Se añade la arista si no existe ya
                if not neighbor in F.neighbors(node):
                    F.add_edge(node, neighbor)
    return F

def add_hidden_variable(F):
    """
    Añade un atributo "internalDegree" a cada nodo de un subgrafo `F`. Este atributo se calcula como la relación 
    entre el grado del nodo y el grado medio del subgrafo.

    Parámetros:
    -----------
    F : networkx.Graph
        Subgrafo en el cual se calculará y añadirá la variable "internalDegree" a cada nodo. 
        Este atributo representa la proporción del grado del nodo en relación al grado medio del subgrafo.

    Retorna:
    --------
    int
        Retorna -1 si el grado medio del subgrafo es 0, lo que indica que no se pudo calcular "internalDegree". 
        No retorna valor si la operación se realiza exitosamente.

    Ejemplo de uso:
    --------------
    result = add_hidden_variable(F)
    if result == -1:
        print("No se pudo calcular 'internalDegree' debido a un grado medio de 0.")
    """
    avg_deg = calc_avg_degree(F)
    if avg_deg != 0:
        dict_hidd_var = {}
        for node in F.nodes():
            dict_hidd_var[node] = F.degree[node] / avg_deg
        nx.set_node_attributes(F, dict_hidd_var, "internalDegree")
    else:
        return -1

def thresh_normalization(G, threshold):
    """
    Aplica la normalización por umbral de grado a un grafo `G` según el método descrito en el artículo 
    "Self-similarity of complex networks and hidden metric spaces" de Angeles et al. El proceso incluye 
    la creación de un subgrafo que contiene solo nodos con un grado superior al umbral especificado, 
    la adición de aristas entre esos nodos, y la incorporación de un atributo "internalDegree" a cada nodo.

    Parámetros:
    -----------
    G : networkx.Graph
        Grafo original del cual se extraerá el subgrafo basado en el umbral de grado. El grafo debe contener
        nodos y aristas entre ellos.

    threshold : int
        Umbral de grado mínimo que un nodo debe tener para ser incluido en el subgrafo. Los nodos que no 
        superen este umbral serán excluidos del subgrafo.

    Retorna:
    --------
    networkx.Graph
        El subgrafo `F` resultante que contiene nodos con grado superior al umbral especificado, junto con
        las aristas correspondientes y el atributo "internalDegree" para cada nodo. Retorna -1 si no hay 
        nodos que cumplan el umbral o si ocurre un error durante el proceso.

    Ejemplo de uso:
    --------------
    F = thresh_normalization(G, threshold=10)
    if F == -1:
        print("No se pudo generar el subgrafo debido a la falta de nodos que cumplan el umbral.")
    """
    # Se añaden solamente los nodos que cumplan el umbral
    F = add_nodes_subgraph(G, threshold)
    
    # Si ya no hay nodos que cumplan el umbral, se acaba el proceso
    if F.number_of_nodes() == 0:
        return -1
    
    # Ahora se añaden las aristas de G de los nodos en el subgrafo F
    F = add_edges_subgraph(G, F)

    # Se añade como variable oculta el grado entre la media del grafo a cada nodo
    if add_hidden_variable(F) == -1:
        return -1
    
    return F

def create_filtered_graph(G, thresh_filt):
    """
    Crea un nuevo grafo a partir de un grafo original, filtrando las aristas según un umbral de peso.

    Dado un grafo `G`, esta función genera un nuevo grafo en el que solo se mantienen las aristas cuyo peso
    es mayor o igual al umbral especificado. Las aristas con peso inferior al umbral son eliminadas.

    Parámetros:
    -----------
    G : networkx.Graph
        El grafo original del cual se va a filtrar.

    thresh_filt : float
        El umbral de peso. Solo se conservarán las aristas cuyo peso sea mayor o igual a este umbral.

    Retorna:
    --------
    networkx.Graph
        Un nuevo grafo que contiene solo las aristas con peso mayor o igual al umbral especificado.
    """
    H = nx.Graph()
    # Añade nodos del grafo original al nuevo grafo
    H.add_nodes_from(G.nodes())

    # Añade aristas que cumplen con el umbral de peso
    for u, v, data in G.edges(data=True):
        if data['weight'] >= thresh_filt:
            H.add_edge(u, v, **data)
    return H
########################################################################
#
# OBTENCIÓN DE MÉTRICAS
#
########################################################################


def convert_keys_to_float(d, recursive=True, tipo="float"):
    """
    Convierte las claves de un diccionario de cadenas a enteros o flotantes. Esta función es útil al procesar
    datos leídos de un archivo JSON donde las claves pueden estar en formato de cadena pero deberían ser números.

    Parámetros:
    -----------
    d : dict
        Diccionario cuyas claves se convertirán a enteros o flotantes. Los valores del diccionario no se modifican 
        salvo que sean otros diccionarios, en cuyo caso se aplica la conversión de forma recursiva.

    recursive : bool, opcional, por defecto True
        Si es True, la conversión se aplica recursivamente a los diccionarios anidados dentro del diccionario principal.

    tipo : str, opcional, por defecto "float"
        Especifica el tipo de conversión a aplicar. Puede ser "int" para enteros o "float" para flotantes. 
        Si el tipo no es "int" ni "float", se asume que se desea convertir a flotante.

    Retorna:
    --------
    dict
        Un nuevo diccionario con las claves convertidas al tipo especificado. Los valores permanecen sin cambios, 
        salvo que sean diccionarios, en cuyo caso también se aplicará la conversión si `recursive` es True.

    Ejemplo de uso:
    --------------
    converted_dict = convert_keys_to_float({"1": "value", "2.5": {"3": "value"}})
    """
    new_dict = {}
    for k, v in d.items():
        # Convierte la clave a entero o a float si es posible
        try:
            if tipo == "int":
                k = int(k)
            else:
                k = float(k)
        except ValueError:
            pass
        
        if recursive:
            # Si el valor es un diccionario, aplica la conversión recursivamente
            if isinstance(v, dict):
                v = convert_keys_to_float(v)
        
        new_dict[k] = v
    return new_dict

def calc_nestedness(G):
    """
    Calcula la métrica de nestedness (NODF) para un grafo dado. Esta métrica mide el grado en que los nodos
    de un grafo están anidados, lo que puede ser útil para evaluar la estructura de las redes bipartitas.

    Parámetros:
    -----------
    G : networkx.Graph
        Grafo para el cual se calculará el índice de nestedness. El grafo debe ser no dirigido y puede 
        tener o no pesos en las aristas.

    Retorna:
    --------
    float
        El valor del índice NODF del grafo, que refleja el nivel de nestedness. Un valor mayor indica un mayor 
        grado de anidamiento en la estructura del grafo.

    Ejemplo de uso:
    --------------
    nodf_score = calc_nestedness(G)
    print(f"NODF Score: {nodf_score}")
    """
    mat = nx.to_numpy_array(G, weight=None)
    mat = mat[~np.all(mat == 0, axis=1)]
    mat = mat[:,~np.all(mat == 0, axis=0)]
    nodf_score = NestednessCalculator(mat).nodf(mat)
    return nodf_score

def get_clust_nest_coefficient(manifestacion, criterio, measures_foler="measures/", datasets_foler="datasets/", graphs_folder="graphs/", write=True, read=True):
    """
    Calcula y devuelve el coeficiente de clustering y el coeficiente de anidamiento para cada hora de una manifestación 
    dada, en función del tipo de red especificado. Los resultados se pueden guardar en un archivo JSON para su reutilización.

    Parámetros:
    -----------
    manifestacion : str
        Nombre de la manifestación para la cual se calcularán las métricas.

    criterio : str
        Determina el tipo de red:
        - 'h': redes con hashtags como nodos.
        - 'u': redes con usuarios como nodos.
        - 'b': redes bipartitas.

    measures_folder : str, opcional, por defecto "measures/"
        Carpeta donde se almacenan los archivos JSON con los resultados calculados.

    datasets_folder : str, opcional, por defecto "datasets/"
        Carpeta donde se encuentran los archivos de datos necesarios para la lectura.

    graphs_folder : str, opcional, por defecto "graphs/"
        Carpeta donde se encuentran los archivos GEXF que representan las redes.

    write : bool, opcional, por defecto True
        Si es True, guarda los resultados calculados en un archivo JSON.

    read : bool, opcional, por defecto True
        Si es True, intenta leer los resultados previamente calculados desde un archivo JSON.

    Retorna:
    --------
    tuple
        Tres listas:
        - `hour_sort` : horas ordenadas de la manifestación.
        - `mod_sort` : coeficientes de modularidad correspondientes a cada hora.
        - `nest_sort` : coeficientes de anidamiento correspondientes a cada hora.

    Ejemplo de uso:
    --------------
    hours, modularities, nestedness = get_clust_nest_coefficient("protest", "u")
    """
    print("Calculando el anidamiento y modularidad de " + manifestacion + " con criterio: " + criterio)
    if criterio == "h":
        name_path = "hashtag"
    elif criterio == "u":
        name_path = "user"

    dict_manif = {}
    path_file = measures_foler + manifestacion + '_' + criterio + '.json'

    if read:
    # Intentamos cargar el archivo que contenga los datos (si existe) si está activa la flag de lectura
        if os.path.exists(path_file):
            try:
                with open(path_file) as f:
                    dict_manif = json.load(f)
                dict_manif = convert_keys_to_float(dict_manif, recursive=False, tipo="int")
            except json.JSONDecodeError:
                dict_manif = {}

    df = read_data(manifestacion, datasets_folder=datasets_foler)
    horas = df["hour"].unique()

    # Se ve que infomación del grafo está ya calculada y, si no, se calcula
    for hora in tqdm(horas):
        hora = int(hora)

        if not hora in dict_manif.keys():
            dict_manif[hora] = {}

        if not ("nestedness" in dict_manif[hora].keys() and "modularity" in dict_manif[hora].keys()):
            if criterio != "b":
                G = nx.read_gexf(graphs_folder + 'nodes_' + name_path + '/' + manifestacion + '/' + str(hora) + '.gexf')
            else:
                G = nx.read_gexf(graphs_folder + 'bipartite/' + manifestacion + '/' + str(hora) + '.gexf')
            if not "nestedness" in dict_manif[hora].keys():
                nestedness = calc_nestedness(G)
                dict_manif[hora]["nestedness"] = float(nestedness)
            
            if not "modularity" in dict_manif[hora].keys(): 
                modularity_louv = nx.community.modularity(G, nx.community.louvain_communities(G, seed=123), weight="weight")
                dict_manif[hora]["modularity"] = modularity_louv

    arr_hour =[]
    arr_nest = []
    arr_mod = []

    for k in dict_manif.keys():
        arr_hour.append(str(k))
        arr_nest.append(dict_manif[k]["nestedness"])
        arr_mod.append(dict_manif[k]["modularity"])

    # Se ordena la información de menor a mayor hora
    data = list(zip(arr_hour, arr_mod, arr_nest))
    data.sort()
    hour_sort, mod_sort, nest_sort = zip(*data)
    hour_sort = list(hour_sort)
    mod_sort = list(mod_sort)
    nest_sort = list(nest_sort)

    # Si esta activa la flag de escritura, se guarda la información en un archivo para no tener que recalular en un futuro
    if write:
        with open(path_file, 'w') as f:
            json.dump(dict_manif, f, indent=2)

    return hour_sort, mod_sort, nest_sort

def calc_avg_clust_coef_by_normalized_internal_degree(G, clust):
    """
    Calcula el coeficiente de clustering promedio para cada valor de grado interno normalizado en un grafo. 

    Dado un grafo `G` y un diccionario que relaciona nodos con su coeficiente de clustering, la función 
    devuelve un diccionario en el que las claves son los grados internos normalizados de los nodos y los 
    valores son la media de los coeficientes de clustering para los nodos que tienen ese grado interno.

    Parámetros:
    -----------
    G : networkx.Graph
        Grafo en el que cada nodo debe tener el atributo "internalDegree" representando su grado interno normalizado.

    clust : dict
        Diccionario donde las claves son nodos del grafo y los valores son los coeficientes de clustering 
        asociados a esos nodos.

    Retorna:
    --------
    dict
        Un diccionario donde las claves son los grados internos normalizados (como valores flotantes) y los 
        valores son la media de los coeficientes de clustering para los nodos con ese grado interno.

    Ejemplo de uso:
    --------------
    avg_clust_coef = calc_avg_clust_coef_by_normalized_internal_degree(G, clust)
    print(avg_clust_coef)
    """
    dict_hid_var_aux = {}
    dict_hid_var = {}
    # Crea un diccionario con cada internal degree como clave y un array con los coeficientes
    # de clusterización de los nodos que tienen dicho internal degree
    arr_int_deg = []

    for node in G.nodes():
        att = G.nodes[node]["internalDegree"]
        if att in dict_hid_var_aux.keys():
            np.append(dict_hid_var_aux[att], clust[node])
        else:
            dict_hid_var_aux[att] = np.array(clust[node])

        arr_int_deg.append(att)

    # Se ordena el diccionario en función de la clave (internal degree) de menor a mayor
    # sorted(dict) devuleve las keys ordenadas
    dict_hid_var_aux_2 = {k: dict_hid_var_aux[k] for k in sorted(dict_hid_var_aux)}

    # Se crea un diccionario con internal degrees como clave y la media de coeficiente de clusterización
    # de los nodos que tienen dicho internal degree como valor
    for key in dict_hid_var_aux_2.keys():
        dict_hid_var[key] = np.average(dict_hid_var_aux_2[key])

    return dict_hid_var

def calc_clust(G, MAX_UMBRAL, measures_path, mode="h", read=True, write=True):
    """
    Calcula la distribución del coeficiente de clustering para una serie de subgrafos generados mediante
    la normalización por umbral iterativa. La función evalúa la evolución de las métricas de clustering 
    a medida que se aplican distintos umbrales de grado a un grafo dado.

    Parámetros:
    -----------
    G : networkx.Graph
        Grafo original sobre el que se calcularán las métricas de clustering.

    MAX_UMBRAL : int
        Número máximo de umbrales a considerar para generar subgrafos.

    measures_path : str
        Ruta base para los archivos de medidas donde se guardan los resultados.

    mode : str, opcional, por defecto "h"
        Modo de cálculo para el tipo de red:
        - 'h': redes con hashtags como nodos.
        - 'u': redes con usuarios como nodos.
        - 'b': redes bipartitas.

    read : bool, opcional, por defecto True
        Si es True, intenta leer los resultados previamente calculados desde archivos JSON.

    write : bool, opcional, por defecto True
        Si es True, guarda los resultados calculados en archivos JSON.

    Retorna:
    --------
    tuple
        Dos diccionarios:
        - `dict_thres_avg_clust` : Diccionario con umbrales como claves y la media de los coeficientes de clustering 
          del subgrafo generado con cada umbral como valores.
        - `dict_norm_int_deg` : Diccionario con umbrales como claves y diccionarios de grados internos normalizados 
          como claves y la media de coeficientes de clustering como valores para cada umbral.

    Ejemplo de uso:
    --------------
    avg_clust, norm_int_deg = calc_clust(G, 10, "path/to/measures", mode="u")
    """
    dict_thres_avg_clust = {}
    dict_norm_int_deg = {}

    measures_path = measures_path + '_' + mode
    if read:
        # Se intenta cargar el archivo donde están los datos
        if os.path.exists(measures_path + '_int_deg.json'):
            try:
                with open(measures_path + '_int_deg.json', 'r') as f:
                    dict_norm_int_deg = json.load(f)
                dict_norm_int_deg = convert_keys_to_float(dict_norm_int_deg)
            except json.JSONDecodeError:
                dict_norm_int_deg = {}

        if os.path.exists(measures_path + '_avg_clust.json'):
            try:
                with open(measures_path + '_avg_clust.json', 'r') as f:
                    dict_thres_avg_clust = json.load(f)
                dict_thres_avg_clust = convert_keys_to_float(dict_thres_avg_clust)
            except json.JSONDecodeError:
                dict_thres_avg_clust = {}

    # Se calculan las sucesivas métricas para cada umbral de grado y se escriben
    for threshold in tqdm(range(MAX_UMBRAL)):
        flag_int_deg = threshold in dict_norm_int_deg.keys()
        flag_avg_clust = threshold in dict_thres_avg_clust.keys()
        # Si ambas son true podemos saltar el paso (ya está calculado previamente)
        if not (flag_int_deg and flag_avg_clust):
            # Se crea el subgrafo basándonos en el threshold seleccionado
            F = thresh_normalization(G, threshold)
            if F == -1:
                # Caso de grafo vacío o grafo inconexo
                # Se escribe la información calculada.
                if write:
                    with open(measures_path + '_int_deg.json', "w") as f:
                        json.dump(dict_norm_int_deg, f, indent=2)
                    with open(measures_path + '_avg_clust.json', "w") as f:
                        json.dump(dict_thres_avg_clust, f, indent=2)

                return dict_thres_avg_clust, dict_norm_int_deg

            if mode == "b":
                clust = nx.algorithms.bipartite.clustering(F)
            else:
                clust  = nx.clustering(F)
            avg_clust = np.mean(np.array(list(clust.values())))
            dict_thres_avg_clust[threshold] = avg_clust

            dict_norm_int_deg[threshold] = calc_avg_clust_coef_by_normalized_internal_degree(F, clust)
        if write:
            with open(measures_path + '_int_deg.json', "w") as f:
                json.dump(dict_norm_int_deg, f, indent=2)
            with open(measures_path + '_avg_clust.json', "w") as f:
                json.dump(dict_thres_avg_clust, f, indent=2)
    return dict_thres_avg_clust, dict_norm_int_deg

def calc_self_sim(hora, MAX_UMBRAL, manifestacion, mode='h', graphs_folder="graphs/", measures_folder="measures/", thresh_filter=5):
    """
    Carga el grafo correspondiente a una hora específica y calcula las métricas de autosimilitud para diferentes umbrales de grado.

    Esta función lee un grafo desde un archivo GEXF basado en los parámetros proporcionados, y luego utiliza la 
    función `calc_clust` para calcular las métricas de autosimilitud como la distribución del coeficiente de clustering 
    para una serie de subgrafos generados por normalización iterativa por umbral.

    Parámetros:
    -----------
    hora : str
        Identificador de la hora específica para la que se debe cargar el grafo.

    MAX_UMBRAL : int
        Número máximo de umbrales a considerar para generar subgrafos y calcular métricas.

    manifestacion : str
        Nombre de la manifestación que se está analizando.

    mode : str, opcional, por defecto 'h'
        Modo de cálculo para el tipo de red:
        - 'h': redes con hashtags como nodos.
        - 'u': redes con usuarios como nodos.
        - 'b': redes bipartitas.
        - 'f': redes filtradas.

    graphs_folder : str, opcional, por defecto "graphs/"
        Ruta a la carpeta que contiene los archivos GEXF de los grafos.

    measures_folder : str, opcional, por defecto "measures/"
        Ruta a la carpeta donde se guardarán o leerán los resultados calculados.

    Retorna:
    --------
    tuple
        Dos diccionarios resultantes de la función `calc_clust`:
        - `dict_thres_avg_clust` : Diccionario con umbrales como claves y la media de los coeficientes de clustering 
          del subgrafo generado con cada umbral como valores.
        - `dict_norm_int_deg` : Diccionario con umbrales como claves y diccionarios de grados internos normalizados 
          como claves y la media de coeficientes de clustering como valores para cada umbral.

    Ejemplo de uso:
    --------------
    avg_clust, norm_int_deg = calc_self_sim("08", 10, "manifestacion1", mode="u")
    print(avg_clust, norm_int_deg)
    """
    if mode == "h":
        path_graph = "nodes_hashtag/"
    elif mode == "u":
        path_graph = "nodes_user/"
    elif mode == "b":
        path_graph = "bipartite/"
    elif mode == "f":
        path_graph = "filtered/" + str(thresh_filter) + '/'
    G = nx.read_gexf(graphs_folder + path_graph  + manifestacion + hora + ".gexf")
    path_measures_hour = measures_folder + manifestacion
    if not os.path.exists(path_measures_hour):
        os.makedirs(path_measures_hour)
    return calc_clust(G, MAX_UMBRAL, path_measures_hour + hora, mode=mode)


########################################################################
#
# OBTENCIÓN DE COEFICIENTES DE LEY DE POTENCIA
#
########################################################################


def get_exp(arr_points):
    """
    Calcula los exponentes de la ley de potencias para una serie de puntos representativos de los grados de los nodos 
    en un grafo. La función ajusta una distribución de ley de potencias a los datos proporcionados y devuelve los 
    resultados del ajuste.

    Parámetros:
    -----------
    arr_points : numpy.ndarray
        Array de puntos que representan los grados de los nodos en el grafo. Los puntos deben ser mayores que cero.

    Retorna:
    --------
    powerlaw.Fit
        Objeto de ajuste de ley de potencias que contiene los resultados del ajuste, incluyendo el exponente estimado 
        y otras métricas asociadas al ajuste de la ley de potencias.

    Ejemplo de uso:
    --------------
    grados_nodos = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    resultados = get_exp(grados_nodos)
    print("Exponente estimado:", resultados.alpha)
    """
    # Se ordenan los puntos de menor a mayor quitando los 0s (producen error al calcular el exponente)
    points_aux = np.sort(arr_points)
    points_aux = points_aux[points_aux != 0]
    points_aux = points_aux[::-1]

    results = powerlaw.Fit(points_aux)
        
    return results

def calc_pdf_points(arr_points):
    """
    Calcula la función de densidad de probabilidad (PDF) para un conjunto de grados o grados normalizados de nodos en un grafo.

    Esta función procesa un array de arrays, donde cada sub-array representa los grados (o grados normalizados) de los nodos en un grafo. 
    Para cada sub-array, la función calcula las probabilidades asociadas a cada grado y devuelve estas probabilidades junto con los grados correspondientes.

    Parámetros:
    -----------
    arr_points : list of numpy.ndarray
        Lista de arrays, donde cada array contiene los grados (o grados normalizados) de los nodos en un grafo. Los grados pueden ser enteros o flotantes.

    Retorna:
    --------
    list of tuple
        Una lista de tuplas, donde cada tupla contiene dos arrays:
        - El primer array (eje X) contiene los grados únicos.
        - El segundo array (eje Y) contiene las probabilidades asociadas a cada grado (frecuencia relativa).

    Ejemplo de uso:
    --------------
    grados = [np.array([1, 2, 2, 3, 3, 3]), np.array([1, 1, 2, 2, 3])]
    pdf_points = calc_pdf_points(grados)
    for degrees, probs in pdf_points:
        print("Grados:", degrees)
        print("Probabilidades:", probs)
    """
    arr_pdf_points = []
    for points in arr_points:
        degrees, counts = np.unique(points, return_counts=True)
        probs = counts / len(points)
        arr_pdf_points.append((degrees, probs))
    return arr_pdf_points

def calc_cdf_points(arr_pdf_points):
    """
    Calcula la función de distribución acumulativa (CDF) a partir de la función de densidad de probabilidad (PDF) para un conjunto de grados de nodos en un grafo.

    Dada una lista de tuplas, donde cada tupla contiene los grados de los nodos y sus respectivas probabilidades (PDF), esta función calcula las probabilidades acumulativas 
    (CDF) y devuelve una lista de tuplas con los grados y sus correspondientes probabilidades acumulativas.

    Parámetros:
    -----------
    arr_pdf_points : list of tuple
        Lista de tuplas, donde cada tupla contiene dos arrays:
        - El primer array (eje X) contiene los grados únicos de los nodos.
        - El segundo array (eje Y) contiene las probabilidades asociadas a cada grado (frecuencia relativa).

    Retorna:
    --------
    list of tuple
        Una lista de tuplas, donde cada tupla contiene dos arrays:
        - El primer array (eje X) contiene los grados únicos.
        - El segundo array (eje Y) contiene las probabilidades acumulativas asociadas a cada grado.

    Ejemplo de uso:
    --------------
    pdf_points = [(np.array([1, 2, 3]), np.array([0.2, 0.5, 0.3]))]
    cdf_points = calc_cdf_points(pdf_points)
    for degrees, cdf in cdf_points:
        print("Grados:", degrees)
        print("CDF:", cdf)
    """
    arr_cdf_points = []
    for pdf_points in arr_pdf_points:
        cdf = np.cumsum(pdf_points[1])
        arr_cdf_points.append((pdf_points[0], cdf))
    return arr_cdf_points

def calc_ccdf_points(arr_cdf_points):
    """
    Calcula la función de distribución acumulativa complementaria (CCDF) a partir de la función de distribución acumulativa (CDF) para un conjunto de grados de nodos en un grafo.

    Dada una lista de tuplas, donde cada tupla contiene los grados de los nodos y sus respectivas probabilidades acumulativas (CDF), esta función calcula las probabilidades acumulativas complementarias 
    (CCDF) y devuelve una lista de tuplas con los grados y sus correspondientes probabilidades acumulativas complementarias.

    Parámetros:
    -----------
    arr_cdf_points : list of tuple
        Lista de tuplas, donde cada tupla contiene dos arrays:
        - El primer array (eje X) contiene los grados únicos de los nodos.
        - El segundo array (eje Y) contiene las probabilidades acumulativas asociadas a cada grado (CDF).

    Retorna:
    --------
    list of tuple
        Una lista de tuplas, donde cada tupla contiene dos arrays:
        - El primer array (eje X) contiene los grados únicos.
        - El segundo array (eje Y) contiene las probabilidades acumulativas complementarias asociadas a cada grado (CCDF).

    Nota:
    -----
    La última entrada de cada CDF se elimina en el resultado final ya que puede deformar el gráfico debido a que la probabilidad complementaria 
    es cero o muy cercana a cero, lo que resulta en una escala logarítmica no útil.

    Ejemplo de uso:
    --------------
    cdf_points = [(np.array([1, 2, 3]), np.array([0.2, 0.5, 1.0]))]
    ccdf_points = calc_ccdf_points(cdf_points)
    for degrees, ccdf in ccdf_points:
        print("Grados:", degrees)
        print("CCDF:", ccdf)
    """
    arr_ccdf_points = []
    for deg_cum in arr_cdf_points:
        ccdf = 1 - deg_cum[1]
        # Se quita el último punto pues, al ser escala logaritimica en los ejes y ser su probablidad
        # complementaria 0 o muy cercana a 0, hace que el grafico quede deformado y no es util
        arr_ccdf_points.append((deg_cum[0][:-1], ccdf[:-1]))   
    return arr_ccdf_points

def calc_degree_distribution(hour, manifestacion, graphs_folder="graphs/", mode="h", measures_folder="measures/", G=None, arr_kt=[0], exp=False, read=True, write=True, norm=False, thresh_filt=5):
    """
    Calcula la distribución de grados de los nodos en un grafo, así como la función de distribución de probabilidad (PDF), la función de distribución acumulativa (CDF) y la función de distribución acumulativa complementaria (CCDF) de los grados de los nodos.
    Opcionalmente, ajusta una ley de potencia a la distribución de grados y calcula el exponente si se solicita.

    Parámetros:
    -----------
    hour : str
        La hora específica del grafo que se está analizando.
    manifestacion : str
        Identificador de la manifestación para la cual se está calculando la distribución.
    graphs_folder : str, opcional
        Ruta al directorio que contiene los archivos de grafos. Por defecto es "graphs/".
    mode : str, opcional
        Modo de los nodos en el grafo ('h' para hashtags, 'u' para usuarios, 'b' para bipartito). Por defecto es "h".
    measures_folder : str, opcional
        Ruta al directorio para almacenar o leer los archivos de medidas. Por defecto es "measures/".
    G : networkx.Graph, opcional
        Objeto de grafo para usar en lugar de cargar uno del archivo. Por defecto es None.
    arr_kt : list, opcional
        Lista de umbrales de grado para aplicar normalización y calcular distribuciones. Por defecto es [0], que indica el grafo original sin normalización.
    exp : bool, opcional
        Si es True, calcula el exponente de la ley de potencia ajustada a la distribución de grados. Por defecto es False.
    read : bool, opcional
        Si es True, intenta leer datos de medidas desde un archivo existente. Por defecto es True.
    write : bool, opcional
        Si es True, guarda las medidas calculadas en un archivo. Por defecto es True.
    norm : bool, opcional
        Si es True, normaliza los grados de los nodos dividiéndolos por la media. Por defecto es False.

    Retorna:
    --------
    plfit : powerlaw.Fit o None
        Objeto que contiene el ajuste de la ley de potencia y el exponente, o None si no se calcula.
    arr_deg_prob : list of tuple
        Lista de tuplas, donde cada tupla contiene dos arrays:
        - El primer array (eje X) contiene los grados únicos de los nodos.
        - El segundo array (eje Y) contiene las probabilidades asociadas a cada grado (PDF).
    arr_deg_comp_cum : list of tuple
        Lista de tuplas, donde cada tupla contiene dos arrays:
        - El primer array (eje X) contiene los grados únicos de los nodos.
        - El segundo array (eje Y) contiene las probabilidades acumulativas complementarias asociadas a cada grado (CCDF).

    Ejemplo de uso:
    --------------
    plfit, pdf_points, ccdf_points = calc_degree_distribution("08", "protest", mode="h", exp=True)
    for degrees, probs in pdf_points:
        print("Grados:", degrees)
        print("PDF:", probs)
    for degrees, ccdf in ccdf_points:
        print("Grados:", degrees)
        print("CCDF:", ccdf)
    """
    if mode == "h":
        path_graph = "nodes_hashtag/"
    elif mode == "u":
        path_graph = "nodes_user/"
    elif mode == "b":
        path_graph = "bipartite/"
    elif mode == "f":
        path_graph = "filtered/" + str(thresh_filt) + '/'
    
    # Se carga el grafo inicial
    G = nx.read_gexf(graphs_folder + path_graph + manifestacion + hour + '.gexf')
        
    arr_points = []
    dict_points = {}
    measures_path = measures_folder + manifestacion + hour + '_' + mode + '_degs_kt.json'
    if read:
        # En los archivos se guardan los grados de los nodos de cada grafo dependiente de kt en bruto, sin sufrir procesos de normalización
        if os.path.exists(measures_path):
            try:
                with open(measures_path, 'r') as f:
                    dict_points = json.load(f)
                dict_points = convert_keys_to_float(dict_points, tipo="int")
            except json.JSONDecodeError:
                pass
    
    # Si no recibe un arr_kt como parametro, se interpreta que es la red original
    for kt in tqdm(arr_kt):
        if not kt in dict_points.keys():
            F = thresh_normalization(G, kt)
            if F != -1:
                points_kt = np.sort(np.array(list(dict(F.degree()).values())).astype(float))
                dict_points[kt] = list(points_kt)
    
        else:
            points_kt = dict_points[kt]
        if norm:
            points_kt = np.array(points_kt) / np.mean(points_kt)

        arr_points.append(points_kt)
    # El exponente solo se va a calcular cuando se reciba un valor de kt
    plfit = None
    if exp:
        # Para calcular el exponente no hay que normalizar
        plfit = get_exp([dict_points[arr_kt[0]]])

    # Puntos de la PDF
    arr_deg_prob = []
    for points in arr_points:
        degrees, counts = np.unique(points, return_counts=True)
        probs = counts / len(points)
        arr_deg_prob.append((degrees, probs))
    
    # Puntos de la CDF
    arr_deg_cum = []
    for deg_prob in arr_deg_prob:
        cum_freq = np.cumsum(deg_prob[1])
        cdf = cum_freq/cum_freq[-1]
        arr_deg_cum.append((deg_prob[0], cdf))

    # Puntos de la CCDF
    arr_deg_comp_cum = []
    for deg_cum in arr_deg_cum:
        ccdf = 1 - deg_cum[1]
        arr_deg_comp_cum.append((deg_cum[0], ccdf))

    if write:
        with open(measures_path, "w") as f:
            json.dump(dict_points, f, indent=2)
    return plfit, arr_deg_prob, arr_deg_comp_cum


########################################################################
#
# FUNCIONES DE REPRESENTACIÓN DE MÉTRICAS
#
########################################################################


def get_all_markers():
    """
    Devuelve una lista de todos los marcadores disponibles en matplotlib.

    Retorna:
    --------
    list
        Una lista de cadenas, donde cada cadena es el nombre de un marcador disponible en matplotlib.
    """
    return [
    '.',  # point marker
    ',',  # pixel marker
    'o',  # circle marker
    'v',  # triangle_down marker
    '^',  # triangle_up marker
    '<',  # triangle_left marker
    '>',  # triangle_right marker
    '1',  # tri_down marker
    '2',  # tri_up marker
    '3',  # tri_left marker
    '4',  # tri_right marker
    's',  # square marker
    'p',  # pentagon marker
    '*',  # star marker
    'h',  # hexagon1 marker
    'H',  # hexagon2 marker
    '+',  # plus marker
    'x',  # x marker
    'D',  # diamond marker
    'd',  # thin_diamond marker
    '|',  # vline marker
    '_',  # hline marker
    'P',  # plus (filled) marker
    'X',  # x (filled) marker
    0,    # tickleft marker
    1,    # tickright marker
    2,    # tickup marker
    3,    # tickdown marker
    4,    # caretleft marker
    5,    # caretright marker
    6,    # caretup marker
    7    # caretdown marker
]