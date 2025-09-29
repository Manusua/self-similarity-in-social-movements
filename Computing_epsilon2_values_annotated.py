"""
Created on Sat Aug  9 11:25:52 2025

This code get a set of networks and their k_T2 values which is the maxi-
mum value of the average degree in the DTR flow (the same as k_T^{max} in the paper )
and return two txt files

one file  contains  4 columns including: Network  epsilon_ccdf  epsilon_knn epsilon_CCO
and the other one have 3 cloumns:  Network Avg_Clustering  epsilon_max


in the code you should replace the value for "Add" to the directory containing the edgelist of your networks 
we suppose that the edgelist of each network saved inside {Network}/{Network}.edge
and "Add2" with the directory you want to save the output txt files 
 
@author: complexity-lab1
"""
import networkx as nx 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from collections import defaultdict
from matplotlib import rc
from matplotlib.colors import to_rgba
import statistics
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from tqdm import tqdm
import time

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Enable LaTeX and add 'amssymb' package to the preamble
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'


#-----------------------------------------------------------------
#It reads the original graph 
# first remove the self loop and extract the giant connected component 
#then remove nodes with degree less than or equal to the Threshold 
def Read_Graph(Add, Network, Threshold):
    
     
    G = nx.read_edgelist(Add + f'{Network}/{Network}.edge', data=False) 
    G.remove_edges_from(nx.selfloop_edges(G))  #remove selfloops
    largest_cc = max(nx.connected_components(G), key=len)
   
    
    if(len(largest_cc) != G.number_of_nodes()):
        print("~~~~~Information about GCC~~~~~~")
        S = [G.subgraph(c).copy() for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        G = S[0]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Remove nodes              
    rm = [n for (n, d) in G.degree() if d <= Threshold]
    G.remove_nodes_from(rm)     
    G.remove_nodes_from(list(nx.isolates(G)))   #remove isolated nodes

    return G

#----------------------
#This function read the original graph
#It removes the self loops and select the giant connected component 
def Read_original_Graph(Add, Network):
    
    G = nx.read_edgelist(Add + f'{Network}/{Network}.edge', data=False) 
    G.remove_edges_from(nx.selfloop_edges(G))  #remove selfloops
    largest_cc = max(nx.connected_components(G), key=len)
   
    
    if(len(largest_cc) != G.number_of_nodes()):
        print("~~~~~Information about GCC~~~~~~")
        S = [G.subgraph(c).copy() for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        G = S[0]

    return G
    
#-----------------------------------------------------------------
#This function get as inputs a network G and a dictionary Res to save the results 
#It retrun Res where the keys are the resclaed degree and the values are the rescaled averge clustering i.e,
#y axis in fig2-a or y-axis in the third column of Fig.S8 in the paper and SI
def Compute_Metric_CCO(G, Res):   
    
    # Create a dictionary of node degrees
    degree_dict = dict(G.degree())
    
    # Filter out nodes with degree 1 and get their clustering coefficients
    filtered_clustering = {
        node: cc for node, cc in nx.clustering(G).items() if degree_dict[node] > 1
    }
    
    # Calculate the average clustering coefficient for the remaining nodes
    if filtered_clustering:
        avg_clustering_coefficient = sum(filtered_clustering.values()) / len(filtered_clustering)
    else:
        avg_clustering_coefficient = 0  # Handle case with no nodes left
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    d = defaultdict(list) #A dictionary in which key=Degree and value = list of nodes with this Degree
    if G.number_of_nodes() != 0:
        Avg_Degree=sum([d for (n, d) in nx.degree(G)]) / float(G.number_of_nodes()) 
    else:
        Avg_Degree = 0
   
    for u in G.nodes():     # add node i to the dictionary d where the key =  k_i
      d[G.degree(u)/Avg_Degree].append(u)
    
    
    for degree in d:    #For each key of the dictionary compute the average clustering coefficient
        #print(degree)
        clustering_coeff = nx.clustering(G, d[degree])
        pp= sum(clustering_coeff.values())/len(clustering_coeff)
        if avg_clustering_coefficient!= 0:
            pp2= pp/avg_clustering_coefficient
        else:
            pp2=0
       
        Res[degree].append(pp2)

    return Res

#-----------------------------------------------------------------
#This function get as inputs a network G and a dictionary Res to save the results 
#It retrun Res where the keys are the resclaed degree and the values are the complementary cumulative
# degree functions i.e., y-axis in the first column of Fig.S8
def Compute_Metric_ccdf(G, Res):
    d = defaultdict(list) #A dictionary in which key=Degree and value = list of nodes with this Degree

    if(G.number_of_nodes() >0):
        Avg_Degree=sum([dd for (n, dd) in nx.degree(G)]) / float(G.number_of_nodes()) 
       
        for u in G.nodes():     # add node i to the dictionary d where the key =  k_i
           d[G.degree(u)/Avg_Degree].append(u)
      
        Deg_G = defaultdict(list) 
        
        
        for degree in d:   
           Deg_G[degree].append(len(d[degree])/G.number_of_nodes())
           
        sorted_Deg_G = {key: Deg_G[key] for key in sorted(Deg_G)}
        cs =  np.cumsum(np.array(list(sorted_Deg_G.values())))
        
        cs2= 1 - cs
        cs3= np.insert(cs2, 0, 1) #add 1 at the top of the list of CCPD
        cs3 = cs3[:-1] #Put this or not does not change the results
        
        for jj, degree in enumerate(sorted_Deg_G):
            Res[degree].append(cs3[jj])
            
    return Res

#-----------------------------------------------------------------
#This function get as inputs a network G and a dictionary Res to save the results 
#It retrun Res where the keys are the resclaed degree and the values are thier rescaled average-nearest neighbor degrees
# i.e., y-axis in the second column of Fig.S8

def compute_Metric_knn(G, Res):
    
    Avg_Degree=sum([dd for (n, dd) in nx.degree(G)]) / float(G.number_of_nodes()) 
    Avg_Degree2=sum([d**2 for (n, d) in nx.degree(G)]) / float(G.number_of_nodes())

    d = defaultdict(list) 
    for u in G.nodes():     # add node i to the dictionary d where the key =  k_i
       d[G.degree(u)/Avg_Degree].append(u)
    Avg_Neigh_Deg= nx.average_neighbor_degree(G)
    
    for degree in d:
        Avg=0
        Temp= d[degree]
        for jj in range(len(Temp)):
            Avg= Avg + Avg_Neigh_Deg[Temp[jj]]
        Avg= Avg / len(Temp)
        Res[degree].append(Avg/(Avg_Degree2/Avg_Degree))
    return Res 
    
#-----------------------------------------------------------------
#It gets the number of bins and the two dictionary including the structural properties from the original network(with kT=2) and the 
#network from DTR and use the values of the sorted keys to indicate the x-axis interval which is in common between
#the two curves 
#it returns an array x_exp indicating the edges of an exponential binning in the common range of the two intervals 
def make_bins(intval1, intval2, nbins):
    
    nonzero_vals1 = [degree for degree in intval1 if degree > 0]
    
    if len(nonzero_vals1) == 0:
        raise ValueError("All values are zero; cannot create logarithmic bins.")

    x0_1 = min(nonzero_vals1)
    xf_1 = max(nonzero_vals1)
    
    
    nonzero_vals2 = [degree for degree in intval2 if degree > 0]
    
    if len(nonzero_vals2) == 0:
        raise ValueError("All values are zero; cannot create logarithmic bins.")

    x0_2 = min(nonzero_vals2)
    xf_2 = max(nonzero_vals2)
    
    x0= max(x0_1, x0_2)
    xf= min(xf_1, xf_2)
    
    xq = (xf/x0) ** (1/nbins)
    

    x_exp= np.zeros(nbins+1)
    x_exp[0]=x0
    
    for i in range(0, nbins):
        x_exp[i+1]=x0 * (xq ** (i+1)) 

    x_exp[len(x_exp)-1]=xf    
    return x_exp
#-----------------------------------------------------------------
#It gets the x_exp from the previous function and a dictionary containing the structrual properties(Res_matrix) 
#and compute the average values of that property inside each bin 
def mean_bins(x_exp, Res_matrix):
    
    mean_values= np.zeros(len(x_exp)-1)
    for i in range(1,(len(x_exp))):  #for each bin
       
        
        values = []  # Initialize values as an empty list for each iteration

        for n, k in enumerate(Res_matrix):
            
            if (i== (len(x_exp)-1)):  #For the last bin
                if x_exp[i-1] <= k <= x_exp[i]:
                    values.append(np.ravel(np.array(Res_matrix[k])))  

                   
            else:
           
                if x_exp[i-1] <= k < x_exp[i]:
                    
                    values.append(np.ravel(np.array(Res_matrix[k])))
                    
        #value will contains all the points inside a bin
        if len(values)>0:  # Check if the list is not empty
            values = np.concatenate(values, axis=0)
            
            mean_values[i-1] = np.mean(values)
    return mean_values

#-----------------------------------------------------------------
#It gets two arrays containing the average of a structural property inside each bin and compute the difference 
#using the formula in algorithm1 of the SI (but only for a pair of original and DTR networks)
def compute_diff(Res_Ref, Res, epsilon_list):
    epsilon= sum([((ccv - ccktv)/ccv)**2 if (ccv > 0)  else (ccv - ccktv)**2 for ccv, ccktv in zip(Res_Ref, Res)])
     
    epsilon2 = epsilon/len(Res)
    epsilon_list.append(epsilon2)

    return epsilon_list    
#-----------------------------------------------------------------  
#It gets a network as an input, removes nodes with degree equal to one 
#and return s the average clustering coefficient 
def Average_Clustering(G):   
    
    # Create a dictionary of node degrees
    degree_dict = dict(G.degree())
    
    # Filter out nodes with degree 1 and get their clustering coefficients
    filtered_clustering = {
        node: cc for node, cc in nx.clustering(G).items() if degree_dict[node] > 1
    }
    
    # Calculate the average clustering coefficient for the remaining nodes
    if filtered_clustering:
        avg_clustering_coefficient = sum(filtered_clustering.values()) / len(filtered_clustering)
    else:
        avg_clustering_coefficient = 0  # Handle case with no nodes left
    
    return avg_clustering_coefficient
      

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#It returns values for k_T chosen in [2, k_T^max ]
def generate_k_loop(k_T2_net, n_max=6, min_dif=3):
    """
    Generate a list of k_T values in the interval of [2, k_T^{max}].
    
    Args:
        k_T2_net (int): k_T^{max}.
        n_max (int): Maximum number of splits (default is 6).
        min_dif (int): Minimum difference between successive k values (default is 3).
    
    Returns:
        loop containing the k_T values 
    """
    n_max -= 1  # Adjust for zero indexing

    loop = []
    Avg_CCO = defaultdict(list)
    AVG_CC_GDO = defaultdict(list)

    Delta_k = max(min_dif, np.floor((int(k_T2_net) - 2) / n_max))
    print("Delta_k:", Delta_k)

    n_curves = max(3, min(n_max, np.floor((int(k_T2_net) - 2) / Delta_k) + 1))
    print("Number of curves:", n_curves)

    for i in range(1, int(n_curves)):
        loop.append(int(2 + (i * Delta_k)))

    if loop[-1] != int(k_T2_net):
        if (int(k_T2_net) - loop[-1] >= min_dif):
            loop.append(int(k_T2_net))
        else:
            loop[-1] = int(k_T2_net)

    return loop


"""
Created on Sat Aug  9 11:25:52 2025

This code get a set of networks and their k_T2 values which is the maxi-
mum value of the average degree in the DTR flow (the same as k_T^{max} in the paper )
and return two txt files

one file  contains  4 columns including: Network  epsilon_ccdf  epsilon_knn epsilon_CCO
and the other one have 3 cloumns:  Network Avg_Clustering  epsilon_max


in the code you should replace the value for "Add" to the directory containing the edgelist of your networks 
we suppose that the edgelist of each network saved inside {Network}/{Network}.edge
and "Add2" with the directory you want to save the output txt files 
 
@author: complexity-lab1
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##List of the netorks and their  k_T^max 
#Networks_list = ["Astrophysics", "bible-nouns",  "PGP", "Internet", "Facebook", "Fb-Friends-Copenhagen", "Int-4896","Solo-BrightKite", "Bitcoin-Trust", "GMP-S.cerevisiaes", "Int-Drosophila", "GI-S.cerevisiae", "Japanese-Words","GMP-Plasmodium", "GMP-Drosophila","GMP-Mus" , "Wikipedia-am",  "MB-R.norvegicus","PPI-rat", "GMP-Celegans", "Int-figeys", "WikiTalk-catalan"]
#k_T2= ["38", "22", "30", "50", "46", "14", "57","49","29",  "94", "13","164", "17", "6" , "26", "7", "35", "8", "9", "9", "8","39" ] 

df = pd.read_csv("epsilon_sq/nat/kts.csv")

Networks_list = list(df["hora"])
k_T2 = list(df["clave_max"])

Add="./epsilon_sq/nat/"
Add2= "./epsilon_sq/nat/results/"


nbins=20

##Open two text files
with open(f"{Add2}/Epsilon_values.txt", "w") as f:
    f.write("Network \t $\\epsilon_{\\ccdf}$  \t  $\\epsilon_{\\knn}$ \t $\\epsilon_{\\cco}$ \n")

with open(f"{Add2}/Epsilon_max_vs_clustering.txt", "w") as f2:
    f2.write("Network \t clustering\\_coeff \t $\\epsilon_{\\max}$ \n")


#For each network in the network list        
for ii, Network in enumerate(Networks_list):            
    Threshold=2  #We consider networks where we remove nodes with degree smaller than or equal to 2 as the original network 
    curve = generate_k_loop(k_T2[ii])
    print(curve)
    #We need to compute the base for comparison which is the cco for the kT=2
    CCO_G_Ref = defaultdict(list) #A dictionary in which key=k_res and value = rescaled Average clustering coefficient for original network 
    ccdf_G_Ref = defaultdict(list) #A dictionary in which key=k_res and value = ccdf for original network 
    knn_G_Ref = defaultdict(list)#A dictionary in which key=k_res and value = rescaled knn for original network 
    
    
    #Read the original network and remove nodes with k_T=2
    G_org = Read_original_Graph(Add, Network)
    Avg_Clustering = Average_Clustering(G_org)
    G = Read_Graph(Add, Network, Threshold)
    
    
    #Compute the structrual properties of the original network 
    Compute_Metric_CCO(G, CCO_G_Ref)
    Compute_Metric_ccdf(G, ccdf_G_Ref)
    compute_Metric_knn(G, knn_G_Ref)
    
    #sort the dictionaries of the structural properties according to their rescaled degree (keys)
    sorted_ccdf_G_Ref = {key: ccdf_G_Ref[key] for key in sorted(ccdf_G_Ref)}       
    sorted_CCO_G_Ref= {key: CCO_G_Ref[key] for key in sorted(CCO_G_Ref)}
    sorted_knn_G_Ref = {key: knn_G_Ref[key] for key in sorted(knn_G_Ref)} 
    
    
    #A list of epsilon values (comaparing each DTR network with original one )
    epsilon_ccdf_G_list = []
    epsilon_CCO_G_list=[]
    epsilon_knn_G_list=[]
    
    #For each value of k_T in the DTR flow 
    for nn, k_T in enumerate(tqdm(curve, desc="Processing")):
        
        print(Network, nn, k_T)
    
        CCO_G = defaultdict(list) #A dictionary in which key=k_res and value = rescaled Average clustering coefficient network in DTR flow 
        ccdf_G = defaultdict(list) #A dictionary in which key=k_res and value = ccdf for the network in DTR flow   
        knn_G = defaultdict(list)  #A dictionary in which key=k_res and value = rescaled knn for the network in DTR flow              
        
        #Read the network and remove nodes with degree smaller than or equal to k_T
        G_kT = Read_Graph(Add, Network, k_T)
        if G_kT.number_of_nodes() != 0:
        #compute the structural properties for the network in DTR flow 
            Compute_Metric_ccdf(G_kT, ccdf_G)
            Compute_Metric_CCO(G_kT, CCO_G)
            compute_Metric_knn(G_kT, knn_G)
        
            #sort the dictionaries of the structural properties according to their rescaled degree (keys)          
            sorted_ccdf_G = defaultdict(list)
            sorted_CCO_G = defaultdict(list)
            sorted_knn_G = defaultdict(list)
            
            sorted_ccdf_G = {key: ccdf_G[key] for key in sorted(ccdf_G)}
            sorted_CCO_G = {key: CCO_G[key] for key in sorted(CCO_G)}
            sorted_knn_G = {key: knn_G[key] for key in sorted(knn_G)}
        
            
            #Do exponential bining in the x-axis range which is in common between the original and DTR networks 
            x_exp_ccdf = make_bins(sorted_ccdf_G, sorted_ccdf_G_Ref, nbins)    
            x_exp_CCO = make_bins(sorted_CCO_G, sorted_CCO_G_Ref, nbins) 
            x_exp_knn = make_bins(sorted_knn_G, sorted_knn_G_Ref, nbins)
            
            
            #Compute the average values of the structural properties inside each bin
            mean_ccdf_G = mean_bins(x_exp_ccdf, sorted_ccdf_G)
            mean_ccdf_G_Ref= mean_bins(x_exp_ccdf, sorted_ccdf_G_Ref)
                                
            mean_CCO_G = mean_bins(x_exp_CCO, sorted_CCO_G)
            mean_CCO_G_Ref= mean_bins(x_exp_CCO, sorted_CCO_G_Ref)
            
            mean_knn_G = mean_bins(x_exp_knn, sorted_knn_G)
            mean_knn_G_Ref= mean_bins(x_exp_knn, sorted_knn_G_Ref)
            
            
            #compute the difference between the curve of the original and the network in DTR flow for all properties      
            epsilon_ccdf_G_list = compute_diff(mean_ccdf_G_Ref, mean_ccdf_G, epsilon_ccdf_G_list)
            epsilon_CCO_G_list = compute_diff(mean_CCO_G_Ref, mean_CCO_G, epsilon_CCO_G_list)  
            epsilon_knn_G_list = compute_diff(mean_knn_G_Ref, mean_knn_G, epsilon_knn_G_list)
            
     
    #Compute the mean for all paris of original and DTR curves     
    epsilon_ccdf = np.mean(epsilon_ccdf_G_list)    
    epsilon_CCO= np.mean(epsilon_CCO_G_list)
    epsilon_knn = np.mean(epsilon_knn_G_list)
    
    print("\n~~~~~~~~~~~~~~~Mean epsilon-square~~~~~~~~~~~~~~~~~~~")
    print(f"Mean epsilon Statistic for ccdf of {Network} is:", epsilon_ccdf)
    print(f"Mean epsilon Statistic for knn of {Network} is:", epsilon_knn)
    print(f"Mean epsilon Statistic for CCO of {Network} is:", epsilon_CCO)
    
    print("============================================================")
    
    #save the results 
    with open(f"{Add2}/Epsilon_values.txt", "a") as f:
        
        f.write(f"{Network} \t {epsilon_ccdf:.4f} \t {epsilon_knn:.4f} \t {epsilon_CCO:.4f} \n")

    epsilon_max =  max(epsilon_CCO, epsilon_knn, epsilon_ccdf)
    with open(f"{Add2}/Epsilon_max_vs_clustering.txt", "a") as f2:
                          
        f2.write(f"{Network} \t {Avg_Clustering:.4f} \t {epsilon_max:.4f} \n")
