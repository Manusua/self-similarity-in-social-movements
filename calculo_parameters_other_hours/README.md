# Parámetros D-Mercator y navegabilidad — ventanas Other TW

Pipeline reproducible para obtener **β**, **μ**, **p_s**, **avg_topo_stretch** y **avg_stretch** de las redes mostradas en los plots `all_self_sim_exponential_o_filtered_*` (columna «Other TW»).

## Ventanas analizadas

| Movimiento | Hora | Ventana | Umbral | Offset respecto CTW |
|------------|------|---------|--------|---------------------|
| NAT | 429600 | 1 | 1 | −24 h |
| 9N | 436989 | 2 | 3 | −48 h |
| CH | 394693 | 2 | 1 | −24 h |

Los edgelists provienen de `graphs/nodes_filtered/{umbral}/{mov}/{ventana}/{hora}.edge`.

## Definiciones

- **p_s** (`success_ratio`): fracción de pares ordenados (s, t) con greedy routing hiperbólico exitoso.
- **avg_topo_stretch**: media de `longitud_greedy / longitud_camino_mínimo` (topológico) sobre pares exitosos.
- **avg_stretch**: media de `suma_distancias_greedy / distancia_hiperbólica_directa` sobre pares exitosos (convención de `navegability.ipynb`).
- **beta, mu**: parámetros del modelo \\(\\mathbb{S}^1\\) inferidos por D-Mercator con `-d 1`.

## Requisitos

- Python 3.10+
- `pandas`, `networkx` (solo para utilidades del repo principal)
- Compilador C++ para D-Mercator (o Docker si falla la compilación local)
- Acceso a red para clonar [networkgeometry/d-mercator](https://github.com/networkgeometry/d-mercator) la primera vez

## Ejecución

Desde la raíz del repositorio:

```bash
python3 calculo_parameters_other_hours/run_all.py
```

Opciones:

```bash
# Recalcular embeddings aunque existan
python3 calculo_parameters_other_hours/run_all.py --force-dmercator

# Solo navegabilidad (embeddings ya generados)
python3 calculo_parameters_other_hours/run_all.py --skip-dmercator
```

## Salidas

```
calculo_parameters_other_hours/
├── inputs/              # edgelists copiados
├── embeddings/          # salidas D-Mercator (.inf_coord, .inf_log, …)
└── results/
    ├── manifest.csv
    ├── parameters_other_hours.csv   # tabla principal
    ├── validation_checks.csv
    ├── dmercator_status.csv
    └── summary.md
```

## Notas

- Los embeddings antiguos en `d-mercator/graphs/` para estas horas pueden estar en formato S² (`Inf.Pos.*`). Este pipeline **regenera** embeddings S¹ homogéneos.
- D-Mercator puede emitir `WARNING: value too high, using beta = …` en redes muy clusterizadas; el embedding sigue siendo válido.
- Para 9N, el embedding previo en `d-mercator/graphs/9n/436989/` correspondía a umbral 1 (908 nodos); aquí se usa umbral 3 (116 nodos), coherente con el plot.
