#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:20:25 2018

@author: muhua
"""

import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tk
import string
#import community
import os
import sys
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', family='serif')
rc('font', size=26)
rc('text', usetex=True)
import matplotlib.pylab as pylab
params = {'axes.labelsize': 22,         
         'xtick.labelsize':20,
         'ytick.labelsize':20} #'legend.fontsize': 'x-large', 'axes.titlesize':'x-large','figure.figsize': (5, 5),
pylab.rcParams.update(params)

#---------------setting format---------------------
color_seq=['#ff3600','#1f77b4', '#5c97e6','#ff850d','#2ca02c',
           '#a82fe0','#aec7e8','#f7b6d2','#dbdb8d','#f26300',
           '#3b00ed','#a60095','#ff690a','#2e00d9','#0055c9']
color_node=['#8DD1C5','#FFFFB5','#BDB9D8','#F98175','#FCB369',
            '#80AFD0','#B2DD6E','#FBCCE4','#D8D8D8','#BA80BB',
            '#E5E5E5']
color_node2=['#2ca02c','#a82fe0','#006DFF','#FF9200',
             '#4CF7C0','#694172','#F4E040','#0677B4','#D08144','#ED9906',
            '#E5E5E5']

mymarker=['s','o','^','v','d','p','*']  
myline= ['-', '--', '-.', ':'] 
dashList = [(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)]   
abcd=list(string.ascii_lowercase)      
#---------------setting format---------------------
#----Mathematica colors ----
m_blue='#5E81B5'
m_green='#8FB131'
m_mustard='#E19C24'
m_tile='#EC6235'
l_grey='#CCCCC6'
# --------------------------

# 1 HYPERBOLIC PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_graph(filename):  
    G1 = nx.Graph()    
    with open(filename) as f:        
        for line in f:
            if not line.lstrip().startswith('#'): 
                  data=line.split() 
                  G1.add_edge(str(data[0]),str(data[1]))
    G1.remove_edges_from(list(nx.selfloop_edges(G1)))    

    return G1
def read_kappa_theta_r(filename):    
    kappa={}
    theta={}
    r={}
    beta=0
    mu=0    
    with open(filename) as f:        
        for line in f:
            if not line.lstrip().startswith('#'): 
                data=line.split() 
                node=str(data[0])
                kappa[node]=float(data[1])
                theta[node]=float(data[2])
                r[node]=float(data[3])
            else:
                data=line.split() 
                if len(data)==4:
                    if data[2]=="beta:":
                        beta=float(data[3])
                    if data[2]=="mu:":
                        mu=float(data[3])
    return beta, mu, kappa, theta, r
def S1toH2(kappa, theta,beta, N, mu):
    R=N/(2*np.pi)
    kappa0=min(kappa.values())
    RH2=2*np.log((2*R)/(mu*kappa0*kappa0))
    r={}
    for i in kappa.keys():        
        r[i]=RH2-2*np.log(kappa[i]/kappa0)
    return r
def connection_prob(thetai,thetaj,kappai,kappaj,R,beta,mu):
    dtheta=np.pi-abs(np.pi-abs(thetai-thetaj))
    xij=R*dtheta/(mu*kappai*kappaj)
    pij=1.0/(1+xij**beta)
    return pij

#def modurarity(G1):      
#    partition1 = community.best_partition(G1)
#    Q1 = community.modularity(partition1,G1)  
#    size=len(set(partition1.values())) # Number of communities
#    return Q1,size,partition1
def change_coord(G,theta, r):
    x={}
    y={}    
    for i in G.nodes():
        x[i]=r[i]*np.cos(theta[i])
        y[i]=r[i]*np.sin(theta[i])      
        
    return x,y
def set_node_size(kappa,a,b):
    '''
    r1=np.asarray(r.values())
    r=30000/(r1**3)    
    rmin=min(r)
    rmax=max(r)    
    node_size=(b-a)*(r-rmin)/(rmax-rmin)+a    
    '''
    node_size={}
    for i in kappa.keys():
        node_size[i]=a+b**np.log(kappa[i])
    return node_size

#-------------------------------------------------------------------------------------
def plot_hypermap(file_edgelist,file_coor,Tc,Scale,fname):
    rx, ry = 3.0, 3.0#3*np.sin(rotation_angle/180.0*np.pi)
    area = rx * ry * np.pi
    theta_angle = np.arange(0, 2 * np.pi + 0.1, 0.1)
    verts = np.column_stack([rx / area * np.cos(theta_angle), ry / area * np.sin(theta_angle)])



    fig= plt.figure(figsize=(8,8))
    gs= gridspec.GridSpec(1,1)#
    ax1 =fig.add_subplot(gs[0,0])  
    
    G=read_graph(file_edgelist)
    beta, mu, kappa, theta, r=read_kappa_theta_r(file_coor)   
    
    N=len(G)    
    R=N/(2*np.pi)    
    
    x,y=change_coord(G,theta,r)
    
    #set node size    
    nodesize=set_node_size(kappa,1,2.5)
    
    #plot edges
    lines=[]
    G_cut=nx.Graph()    
    for u,v in G.edges():    
        pij=connection_prob(theta[u],theta[v],kappa[u],kappa[v],R,beta,mu)
        if pij>Tc:
            lines.append([(x[u], y[u]), (x[v], y[v])])
            G_cut.add_edge(u,v)
    lc = LineCollection(lines, colors=l_grey,linewidths=0.3,linestyles='solid',zorder=1)
    ax1.add_collection(lc)
    # plot nodes
    for i in G_cut.nodes():
        a=[x[i]]
        b=[y[i]]        
        ax1.scatter(a,b, marker='o',color=color_seq[2],linewidths=0.1,edgecolor='k',

                 s=Scale*nodesize[i],zorder=2)

    # Draw a circle 
    r_cut={}
    for i in G_cut.nodes():
        r_cut[i]=r[i]
    rmax=max(r_cut.values())    
    theta_angle=np.linspace(0, np.pi, 100)
    xc=[]
    y_upper = []
    y_lower = []
    for thetax in theta_angle:     
        xc.append(rmax*np.cos(thetax))
        y_upper.append(rmax*np.sin(thetax))
        y_lower.append(-rmax*np.sin(thetax) )
    ax1.fill_between(xc, y_upper, y_lower,linewidth=1.5, facecolor=color_seq[8], 
                     edgecolor='k',zorder=-5, alpha=0.2)

    ax1.set_ylim(-rmax,rmax)
    ax1.set_xlim(-rmax,rmax)
    ax1.axis('off')
    ax1.axis('equal')
    #fig.subplots_adjust(top=0.97, bottom=0.08, left=0.05, right=0.95)
    print("Guardando en",fname+'.png')
    plt.savefig(fname+'.png',bbox_inches='tight',dpi=450)#fig.dpi 
    plt.show()

def main():
    # print command line arguments
#    for arg in sys.argv[1:]:
#        print arg
    fname, ext = os.path.splitext(sys.argv[1])
    plot_hypermap(str(sys.argv[1]),str(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),str(fname))
if __name__ == "__main__":
    main()


