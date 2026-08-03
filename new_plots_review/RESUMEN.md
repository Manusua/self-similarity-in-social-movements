# Resumen de la revisión integral (`new_plots_review/`)

Este directorio contiene **todos los análisis, figuras, tablas y borradores de respuesta** generados para atender los comentarios de revisores y las notas de reunión.

## Cómo reproducir

Desde la raíz del repositorio:

```bash
python3 new_plots_review/run_all.py
```

O ejecutar celda a celda el notebook [`new_plots.ipynb`](new_plots.ipynb).

---

## 1. Fig. 2 — series temporales en hora local

**Qué se hizo**
- Regeneración de los paneles temporales (actividad, modularidad/anidamiento, \(\epsilon^2_{\mathrm{cco}}\)) en **hora local**:
  - NAT / 9N → `America/Argentina/Buenos_Aires`
  - CH → `Europe/Paris`
- Línea CTW en **negro discontinua** (`-.`), grosor moderado (`linewidth=1.8`).
- **Modularidad** (eje Y izquierdo) y **nestedness** (eje Y derecho) en el mismo panel, sin leyendas internas.
- Ventanas gris/naranja según `config.ANNOTATION_WINDOWS`.
- Notación unificada: \(\epsilon^2_{\mathrm{cco}}\).

**Figuras**
- `figures/fig2_nat_local_time.png`
- `figures/fig2_9n_local_time.png`
- `figures/fig2_ch_local_time.png`
- `figures/fig2_combined_local_time.png` (paneles a–i)

**Módulo:** `temporal_plots.py`

---

## 2. Suavizado temporal (MA, mediana, Savitzky–Golay)

**Qué se hizo**
- Comparación de cuatro filtros sobre \(\epsilon^2_{\mathrm{cco}}\):
  - crudo
  - media móvil (3)
  - mediana móvil (3)
  - Savitzky–Golay (ventana 5, orden 2)
- Correlaciones guardadas por manifestación.

**Conclusión operativa:** las curvas son muy similares; se mantiene la media móvil N=3 del manuscrito. El suavizado **reduce fluctuaciones de muestreo, no elimina outliers**.

**Resultados**
- `results/smoothing_{nat,9n,ch}.csv`
- `results/smoothing_correlations_{nat,9n,ch}.csv`
- `figures/smoothing_{nat,9n,ch}.png`

---

## 3. Comunidades: Q fija, AMI y entropía

**Qué se hizo**
- Sobre grafos **GEXF hashtag ponderados** (incluye nodos aislados):
  - Partición Louvain fijada en la CTW (`seed=42`).
  - \(Q(G_t, \mathcal{C}_{\mathrm{CTW}})\) vs \(Q\) reoptimizado.
  - AMI y co-miembros de pares en el universo CTW.
- Comparación con AMI binario existente (`measures/ami_louvain_vs_critica.csv`).
- **Entropía normalizada de tamaños de comunidad** para ventanas 1–8 h (diagnóstico, no selector único).

**Resultados**
- `results/community_fixed_{nat,9n,ch}.csv`
- `results/window_entropy_diagnostic.csv`
- `figures/community_fixed_{nat,9n,ch}.png`

**Módulo:** `community_analysis.py`, `ami_utils.py`

---

## 4. \(\epsilon^2\) canónico (degree / clustering / knn)

**Qué se hizo**
- Se tomó como **definición canónica** el pipeline de `Computing_epsilon2_values_annotated.py`, refactorizado en `epsilon_metrics.py`:
  - GCC, referencia \(k_T=2\), 20 bins exponenciales, normalización CCO.
- **Regresión numérica** frente a `epsilon_sq/*/results/Epsilon_values.txt` (coincidencia ~1e-4).
- Valores CTW reportados por componente:

| Mov. | \(\epsilon^2_{\mathrm{ccdf}}\) | \(\epsilon^2_{\mathrm{knn}}\) | \(\epsilon^2_{\mathrm{cco}}\) |
|------|----------------------------------|-------------------------------|-------------------------------|
| NAT  | 0.2722 | 0.1926 | 0.1746 |
| 9N   | 0.3072 | 0.1683 | 0.1344 |
| CH   | 0.3453 | 0.3341 | 0.2082 |

**Resultados**
- `results/epsilon_ctw_summary.csv`
- `results/epsilon_regression_check.csv`
- `figures/epsilon_collapse_{nat,9n,ch}.png`

---

## 5. Modelos nulos (configuration model)

**Qué se hizo**
- 10 réplicas degree-preserving por CTW.
- Contraste de \(\epsilon^2\), clustering y longitud de camino observada vs nulo.

**Resultados**
- `results/null_model_{nat,9n,ch}.csv`
- `figures/null_model_{nat,9n,ch}.png`

---

## 6. Non-CTW, D-Mercator y navegabilidad

**Selección non-CTW** (`results/nonctw_selection.csv`)

| Mov. | CTW | non-CTW | actividad rel. | \(\epsilon^2_{\mathrm{cco}}\) non-CTW |
|------|-----|---------|----------------|---------------------------------------|
| NAT  | 429624 | 429600 | 71% | 0.5168 |
| 9N   | 437037 | 437075 | 85% | 0.7252 |
| CH   | 394717 | 394701 | 94% | 0.5286 |

Regla: fuera de ±12 h, actividad ±20% (±30% si necesario), máximo \(\epsilon^2_{\mathrm{cco}}\).

**Embeddings**
- Reutilizados CTW NAT/9N existentes.
- **Calculado CH 394717** (canónico) con D-Mercator v0.9 local (`vendor/d-mercator/mercator`).
- Sensibilidad CH **394718** conservada.
- non-CTW calculados para las tres manifestaciones.

**Navegabilidad (greedy routing, todos los pares)**

| Mov. | Ventana | success | topo stretch |
|------|---------|---------|--------------|
| NAT  | CTW | 0.983 | 1.023 |
| NAT  | non-CTW | 1.000 | 1.023 |
| 9N   | CTW | 0.950 | 1.023 |
| 9N   | non-CTW | 0.995 | 1.018 |
| CH   | CTW (394717) | **0.992** | 1.021 |
| CH   | non-CTW | 0.873 | 1.008 |

**Interpretación:** en CH la CTW canónica es claramente más navegable que la ventana non-CTW emparejada; en NAT/9N la diferencia es más sutil (non-CTW seleccionada por alto \(\epsilon^2\), no por baja navegabilidad).

**Resultados**
- `results/embedding_inventory.csv`
- `results/navigability_summary.csv`
- `figures/navigability_ctw_vs_nonctw.png`
- `figures/embedding_pconn_*` (probabilidad de conexión SI)

**Módulos:** `dmercator_utils.py`, `navigability_utils.py`, `embedding_figures.py`

---

## 7. Respuestas a revisores

**Archivos listos para copiar al informe de respuesta (inglés):**
- `results/reviewer_response_matrix.csv`
- `results/reviewer_response_matrix.json`

Cubre: Fig. 2, hora local, notación \(\epsilon^2\), suavizado, Q fija/AMI, nulos, embeddings non-CTW, navegabilidad, Savitzky–Golay, criterio de ventana.

---

## Estructura de archivos

```
new_plots_review/
├── config.py                 # Constantes globales (CTW, zonas, rutas)
├── temporal_plots.py         # Fig. 2 + suavizado
├── community_analysis.py     # Q fija, AMI, entropía
├── epsilon_metrics.py        # Pipeline ε² canónico
├── null_models.py            # Nulos + figuras ε²
├── dmercator_utils.py        # Selección non-CTW + D-Mercator
├── navigability_utils.py     # Greedy routing
├── embedding_figures.py      # Figuras SI de embeddings
├── generate_responses.py     # Matriz de respuestas
├── run_all.py                # Pipeline completo
├── new_plots.ipynb           # Notebook reproducible
├── RESUMEN.md                # Este archivo
├── figures/                  # Todas las figuras
├── results/                  # CSV/JSON
├── embeddings/               # Embeddings nuevos (CH 394717 + non-CTW)
└── vendor/d-mercator/        # D-Mercator oficial clonado y compilado
```

---

## Cambios recomendados en manuscrito / SI

1. Sustituir **UTC** por **hora local** en captions de Fig. 2 y SI.1–SI.2.
2. Unificar notación a \(\epsilon^2_{\mathrm{cco}}\), \(\epsilon^2_{\mathrm{ccdf}}\), \(\epsilon^2_{\mathrm{knn}}\); eliminar “test”.
3. Caption Fig. 2: nestedness en **verde**, línea CTW **negra** discontinua; paneles (a–c), (d–f), (g–i).
4. Añadir párrafo: suavizado para **fluctuaciones**, no outliers.
5. Añadir análisis Q fija + AMI en SI.
6. Moderar claims de difusión usando navegabilidad CTW vs non-CTW.
7. Corregir SI Sec. III: ventanas non-CTW (no copiar CTW).
8. Reportar \(\epsilon^2_{\mathrm{cco}}\) en Fig. 3 / SI.3.
9. CH canónico = **394717**; 394718 solo sensibilidad.

---

## Limitaciones

- NAT non-CTW (429600) produce red muy pequeña tras filtrado GCC (18 nodos); interpretar con cautela.
- AMI binario histórico (edgelist) difiere del pipeline ponderado GEXF; ambos se reportan.
- D-Mercator requiere compilador C++ (`vendor/d-mercator/mercator`); Docker no es necesario si se compila localmente.
