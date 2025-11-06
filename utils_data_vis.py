import networkx as nx
import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np
import powerlaw
import csv
import datetime


def create_csv_weighted(input_file, output_file):
    df = pd.read_csv(input_file, sep= ',')
    df['weight'] = df.groupby(['user', 'hashtag', 'hour'])['user'].transform('count')
    result = df.drop_duplicates(subset=['user', 'hashtag', 'hour'])
    result.to_csv(output_file, sep=' ',index=False)
    return

def get_num_users_hashtags(df, hour_window=1):
    num_users = []
    num_hashtags = []
    for hour in np.sort(df["hour"].unique())[::hour_window]:
        conditions = (df["hour"] == hour)
        for step in range(1, hour_window):
            conditions |= (df["hour"]  == hour + step)
        df_hour = df[conditions] 
        num_users.append(len(df_hour["user"].unique()))
        num_hashtags.append(len(df_hour["hashtag"].unique()))
    return num_users, num_hashtags

def get_num_nodes_edges(df_hour, node_type, manifestacion, hour_window=1, graphs_folder="graphs/"):
    num_edges = []
    num_nodes = []
    for hour in tqdm(df_hour):
        G = nx.read_gexf(graphs_folder + f"nodes_{node_type}/{manifestacion}/" + str(hour_window) + '/' + str(hour) + '.gexf')
        num_edges.append(G.number_of_edges())
        num_nodes.append(G.number_of_nodes())
    return num_nodes, num_edges


########################################################################
#
# FUNCIONES DE PLOTS
#
########################################################################

def plot_num_users_hashtags(ax, hour_dt, hour_window, num_users, num_hashtags, title=None, inicio=0, final=None, legend=False, HORA_CRITICA=None, with_avg = False):
    if title:
        ax.set_title(title, fontsize=20)
    if not final:
        final = len(hour_dt)
    if with_avg:
        ax.plot(hour_dt[int(inicio/hour_window):int(final/hour_window)], num_users[int(inicio/hour_window):int(final/hour_window)], label="Número de usuarios únicos\nMedia de usuarios por hora: " +str(round(np.mean(num_users), 1)), color="orange")
        ax.plot(hour_dt[int(inicio/hour_window):int(final/hour_window)], num_hashtags[int(inicio/hour_window):int(final/hour_window)], label="Número de hashtags únicos\nMedia de hashtags por hora: " +str(round(np.mean(num_hashtags), 1)), color="magenta")
    else:
        ax.plot(hour_dt[int(inicio/hour_window):int(final/hour_window)], num_users[int(inicio/hour_window):int(final/hour_window)], label="Unique users", color="orange")
        ax.plot(hour_dt[int(inicio/hour_window):int(final/hour_window)], num_hashtags[int(inicio/hour_window):int(final/hour_window)], label="Unique Hashtags", color="magenta")
    if HORA_CRITICA:
        ax.axvline(x=datetime.datetime.fromtimestamp(int(HORA_CRITICA)*3600, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H"), color="green", ls="--", label="Fecha identificada como crítica: " + str(datetime.datetime.fromtimestamp(int(HORA_CRITICA)*3600, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H")) +' (UTC)')
    ax.get_xaxis().set_visible(False)
    if legend:
        ax.legend(loc="best", prop={'size': 12})

def plot_mod_nestedness(ax, hour_dt, hour_window, mod_sort, nest_sort, inicio=0, final=None, legend=False, tick_split=30, HORA_CRITICA=None):
    ax.plot(hour_dt[int(inicio/hour_window):int(final/hour_window)], mod_sort[int(inicio/hour_window):int(final/hour_window)], label="Modularity")
    ax.plot(hour_dt[int(inicio/hour_window):int(final/hour_window)], nest_sort[int(inicio/hour_window):int(final/hour_window)], label="Nestedness")
    if HORA_CRITICA:
        ax.axvline(x=datetime.datetime.fromtimestamp(int(HORA_CRITICA)*3600, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H"), color="green", ls="--")
    ax.get_xaxis().set_visible(True)
    ax.set_xticks(hour_dt[int(inicio/hour_window):int(final/hour_window):tick_split])
    ax.set_xticklabels(hour_dt[int(inicio/hour_window):int(final/hour_window):tick_split])
    ax.tick_params(axis='x', rotation=70)
    if legend:
        ax.legend(loc="best", prop={'size': 12})

def plot_pdf(ax, arr_deg_prob, arr_kt_plot, title=None, ylabel="P(X>x)", ylim=(0.0002, 1.05), xlim=None, marker= "x", alpha=0.7, dot_size=4):
    if title:
        ax.set_title(title, fontsize=20)
    for index, points in enumerate(arr_deg_prob):
        ax.scatter(points[0], points[1], marker=marker, s=dot_size, alpha=alpha, label=f"$k_t :$ {str(arr_kt_plot[index])}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20)


def plot_avg_deg(ax, dict_hora, limit_kt, axvline=None, xlabel=None, ylabel=None, alpha=0.7, dot_size=6, ylim=None):
    ax.set_facecolor('xkcd:white')
    for spine in ax.spines.values():
        spine.set_visible(True)  # Asegurar que sea visible
        spine.set_linewidth(0.5)   # Grosor del borde (opcional)
        spine.set_color("black") 
    ax.get_xaxis().set_visible(True)
    ax.plot(list(dict_hora.keys())[0:limit_kt], list(dict_hora.values())[0:limit_kt], alpha=alpha)
    ax.scatter(list(dict_hora.keys())[0:limit_kt], list(dict_hora.values())[0:limit_kt], s=dot_size, alpha=alpha)
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if axvline:
        ax.axvline(axvline, ls="--")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=20)


def plot_clust_ss(ax, arr_kt_plot, dict_norm_int_deg, xlim=None, ylim=(0.1, 1.05), legend=True, marker="x", alpha=0.7, ylabel=None, xlabel=None, sep_points=1, loc=None, linewidth=0.6, s=4):
    
    for kt in arr_kt_plot[:]:
        if kt in dict_norm_int_deg.keys():
            points_x = list(dict_norm_int_deg[kt].keys())[::sep_points]
            points_y = list(dict_norm_int_deg[kt].values())[::sep_points]
            # Se quita el 0 para que la visualizacións se más clara
            if float(0) in dict_norm_int_deg[kt].keys():
                points_x = points_x[1:]
                points_y = points_y[1:]

            points_y, indexes = calcular_quitar_ceros(points_y)
            points_x, _ = quitar_ceros(points_x, indexes)
    
            ax.plot(points_x, points_y, alpha=alpha, linewidth=linewidth)
            ax.scatter(points_x, points_y, alpha=alpha, s=s, marker=marker, label=f'$k_T: {kt}$')        
    if ylabel:  
        ax.set_ylabel(ylabel, fontsize=20)
        #ax.set_ylabel("$\\overline{c(Q_n(k_T))}$", fontsize=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=20)
        #ax.set_xlabel("$Q_n(k_T)$", fontsize=20)
    ax.set_xscale('log')  
    ax.set_yscale('log')  
    if ylim:    
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if legend:
        if loc:
            ax.legend(loc=loc, prop={'size': 12})
        else:
            ax.legend(prop={'size': 12})


########################################################################
#
# FUNCIONES DE PLOTS AUXILIARES
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

def quitar_ceros(lista, indices):
    indices.sort(reverse=True)
    for indice in indices:
        if 0 <= indice < len(lista):
            del lista[indice]  
    return lista, indices

def calcular_quitar_ceros(lista):
    indices = []
    for indice, valor in enumerate(lista):
        if valor == float(0):
            indices.append(indice)
    return quitar_ceros(lista, indices)
