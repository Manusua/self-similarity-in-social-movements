

from xml.etree import ElementTree as ET
from typing import Dict


def anonimizar_gexf(ruta_entrada: str, ruta_salida: str, inicio_id: int = 0) -> None:
    """
    Lee un archivo GEXF, anonimiza los nodos asignándoles IDs numéricos incrementales
    y guarda un nuevo GEXF manteniendo las mismas aristas y pesos.

    Parámetros:
        ruta_entrada (str): Ruta al archivo GEXF original.
        ruta_salida (str): Ruta donde se guardará el GEXF anonimizado.
        inicio_id (int): ID inicial para la numeración (por defecto 0).

    Notas:
        - Se reemplaza el atributo 'id' de cada nodo.
        - Si el nodo tiene atributo 'label', también se reemplaza por el nuevo ID.
        - Se actualizan 'source' y 'target' de las aristas para que apunten a los nuevos IDs.
        - Los pesos ('weight') y el resto de atributos se conservan.
    """

    def localname(tag: str) -> str:
        """Devuelve el nombre local de una etiqueta XML, ignorando el namespace."""
        return tag.split("}", 1)[-1] if "}" in tag else tag

    # Parsear XML
    tree = ET.parse(ruta_entrada)
    root = tree.getroot()

    # Registrar namespace por defecto (si existe) para evitar prefijos raros como ns0
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}", 1)[0][1:]
        ET.register_namespace("", ns_uri)

    # Encontrar nodos y aristas
    nodos = [elem for elem in root.iter() if localname(elem.tag) == "node"]
    aristas = [elem for elem in root.iter() if localname(elem.tag) == "edge"]

    if not nodos:
        raise ValueError("No se encontraron nodos en el archivo GEXF.")

    # Crear mapeo: id_original -> id_nuevo
    mapeo_ids: Dict[str, str] = {}
    siguiente_id = inicio_id

    for nodo in nodos:
        id_original = nodo.get("id")
        if id_original is None:
            raise ValueError("Se encontró un nodo sin atributo 'id'.")

        if id_original in mapeo_ids:
            raise ValueError(f"ID de nodo duplicado en el GEXF: {id_original}")

        id_nuevo = str(siguiente_id)
        siguiente_id += 1

        mapeo_ids[id_original] = id_nuevo

        # Reemplazar id del nodo
        nodo.set("id", id_nuevo)

        # Reemplazar label si existe (para anonimizar el "nombre")
        if "label" in nodo.attrib:
            nodo.set("label", id_nuevo)

    # Actualizar source/target de las aristas
    for arista in aristas:
        source = arista.get("source")
        target = arista.get("target")

        if source is None or target is None:
            raise ValueError("Se encontró una arista sin 'source' o 'target'.")

        if source not in mapeo_ids:
            raise ValueError(f"La arista referencia un nodo source inexistente: {source}")
        if target not in mapeo_ids:
            raise ValueError(f"La arista referencia un nodo target inexistente: {target}")

        arista.set("source", mapeo_ids[source])
        arista.set("target", mapeo_ids[target])

        # El peso ('weight') y demás atributos se conservan tal cual.

    # Guardar archivo resultante
    tree.write(ruta_salida, encoding="utf-8", xml_declaration=True)

graph_origin_1 = "graphs/nodes_filtered/1/ch/2/394645.gexf"
graph_dest_1 = "graphs/nodes_filtered_anonymized/1/ch/2/394645.gexf"


anonimizar_gexf(graph_origin_1, graph_dest_1)

graph_dest_edge_list_1 = "graphs/nodes_filtered_anonymized/1/ch/2/CH_other_TW_edges.txt"

import networkx as nx

G = nx.read_gexf(graph_dest_1)
with open(graph_dest_edge_list_1, "wb") as f:
    nx.write_edgelist(G, f, data=False)



graph_origin_2 = "graphs/nodes_filtered/1/ch/2/394717.gexf"
graph_dest_2 = "graphs/nodes_filtered_anonymized/1/ch/2/394717.gexf"


anonimizar_gexf(graph_origin_2, graph_dest_2)

graph_dest_edge_list_2 = "graphs/nodes_filtered_anonymized/1/ch/2/CH_CTW_edges.txt"

import networkx as nx

G = nx.read_gexf(graph_dest_2)
with open(graph_dest_edge_list_2, "wb") as f:
    nx.write_edgelist(G, f, data=False)