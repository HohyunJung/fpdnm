
"""
@author: Hohyun Jung
Fitness-Popularity Dynamic Network Model
Jung, H., Lee, J., Lee, N., and Kim, S., (2018) "Comparison of Fitness and Popularity: Fitness-Popularity Dynamic Network Model," Journal of Statistical Mechanics: Theory and Experiment.
Assumption: invariant fitness

Usage: 

    fp = FPDNM(data=Data, datatype=2, sampling_method="initial", n_eff=300, beta0=[-5.0000, 1.0000, 0.0000], M=1000)
    fp.EM_(max_it=150, tol=0.00001, disp=True)
"""

from __future__ import division
from numpy import sum
from numpy.linalg import inv, norm
from scipy.optimize import minimize
from copy import deepcopy
import numpy as np
from arspy.ars import adaptive_rejection_sampling
import time
import networkx as nx

class FPDNM():
    def __init__(self, data, beta0=[-4.0,0.1,0.0], datatype=1, m=500, po_type="indegree", po_transtype=1, r=1, n_eff=None, M=None, 
                 sampling_method="default", nodelist=None, fi_dist="normal", descriptions=True, ofBdry=200, yAdd=0.1):
        self.pareto_alpha = 3             # see Pareto distribution wiki page
        self.gamma_k = 2                  # see Gamma distribution wiki page
        self.ofBdry = ofBdry              # prevents overflow error in exponential function
        self.yAdd = yAdd                  # prevents poor estimate
        self.data = data                  # data format: list of dictionaries, {'nodeList': nodeList, 'indegDict': indegDict, 'edgeList': edgeList}}, t=0,1,...,T-1.
        self.beta0 = np.asarray(beta0)
        self.datatype = datatype          # 1: message,  2: friendship(non-edge-deletion network)
        self.po_type = po_type            # See popularity_t.
        self.po_transtype = po_transtype  # See po_transformation.
        self.r = r
        self.n_eff = n_eff                # Number of effective sender nodes per each receiver node. n_j^t.
        self.M = M                        # Number of node samples to be considered
        self.sampling_method = sampling_method    # Decide how the nodes are sampled. Go nodes_() function for details.
        self.nodelist = nodelist
        self.fi_dist = fi_dist            # Fitness distribution (prior), options: "normal", "log-exponential", "pareto" (Warning: "pareto" is not adequate because of non-log-concativity.)
        self.ars_a, self.ars_b, self.ars_domain = self.ars_params()           # ARS function's parameters
        self.nodes = self.nodes_()        # Select M nodes if M is specified.
        self.n_pa = len(self.beta0)
        self.m = m                        # Number of fitness samples used in EM algorithm
        self.T = len(data)
        self.n = self.n_()
        self.y = self.y_()
        self.po = self.popularity_()
        self.z = dict()
        self.beta = None
        self.betaCov = None
        self.beta_list = list()         
        if descriptions == True :
            print("Number of nodes: %d." %(len(self.data[0]['nodeList'])))
            print("Number of edges (t=0, 1, ... , T-1): ", end=' ')
            for t in range(self.T) : print(len(self.data[t]['edgeList']), end=' ')
            print("\nNumber of being used(or sampled) nodes: %d." %(len(self.nodes)))
            print("Total number of in-degrees over being used nodes (t=0, 1, ... , T-1): ", end=' ')
            for t in range(self.T) :
                idd = self.data[t]['indegDict']
                print(sum([idd[x] for x in self.nodes]), end=' ')
            print("")

    def nodes_(self):
        if self.sampling_method == "default":                 # "default": consider all nodes.
            nodes_all = self.data[0]['nodeList']
            if self.M == None : 
                return nodes_all
            else :
                return list(np.random.choice(nodes_all, self.M, replace=False))
        elif self.sampling_method == "initial":               # "initial": consider initial graph's nodes (having at least one in- or out- degree.)
            initial_edges = self.data[0]['edgeList']
            initial_nodes = list(set().union([x for (x,y) in initial_edges], [y for (x,y) in initial_edges]))
            if self.M == None : 
                return initial_nodes
            else :
                return list(np.random.choice(initial_nodes, self.M, replace=False))
        elif self.sampling_method == "designate":               # "designate": use designated nodelist
            return self.nodelist
        else :
            raise NameError('Undefined node sampling method. Check sampling_method.')

    def mf(self, ftn):               
        def f(x):
            return -ftn(x)
        return f

    def y_transform(self, y, n):
        if y < n :      y_new = y + self.yAdd
        elif y == n :   y_new = n - self.yAdd
        else :          
            #y_new = y
            raise NameError('Check the response variable. Consider increasing n_eff.')
        return y_new
        
    def n_t(self, t):                     # n dictionary {j:n_j^t}
        N = len(self.data[t]['nodeList'])
        if self.datatype == 1 :
            n_t = {x:N-1 for x in self.nodes}
        elif self.datatype == 2 :
            indeg_prev = self.data[t-1]['indegDict']
            n_t = {x:N-1-indeg_prev[x] for x in self.nodes}
        else :
            raise NameError('Undefined datatype.')
        return n_t

    def n_(self):                         # length T-1 n's list: n=[n[1], ..., n[T-1]]
        n = list()                        # Caution: indexing
        for t in range(1,self.T):
            n.append(self.ny_t(t)[0])
        return n

    def n_idx(self, t):                   # Code n[t] actually gives t+1 time of n. Input: real time, Output : index of list
        return t-1

    def y_t(self, t):                     # y dictionary {j:y_j^t}
        if self.datatype == 1 :
            y_t = {x:self.data[t]['indegDict'][x] for x in self.nodes}
        elif self.datatype == 2 :
            y_t = {x:self.data[t]['indegDict'][x] - self.data[t-1]['indegDict'][x] for x in self.nodes}
        else :
            raise NameError('Undefined datatype.')
        return y_t

    def y_(self):                         # length T-1 y's list: y=[y[1], ..., y[T-1]]
        y = list()                        # Caution: indexing
        for t in range(1,self.T):
            y.append(self.ny_t(t)[1])
        return y
		
    def y0_(self):                       # length T-1 y0's list: y=[y0[1], ..., y0[T-1]]
        y0 = list()                      # Caution: indexing
        for t in range(1,self.T):
            y0.append(self.n0n1y0y1_(t)[2])
        return y0

    def y_idx(self, t):                   # Code y[t] actually gives t+1 time of y. Input: real time, Output: index of list
        return t-1

    def ny_t(self, t):
        n = self.n_t(t)
        y = self.y_t(t)
        if self.n_eff == None :
            n_new = {x:max(self.r*n[x], y[x]) for x in n}
            y_new = {x:self.y_transform(y[x], n_new[x]) for x in y}
        else :
            n_new = {x:max(self.n_eff, y[x]) for x in n}
            y_new = {x:self.y_transform(y[x], n_new[x]) for x in y}
        return n_new, y_new

    def po_transformation(self, Dict):   
        if self.po_transtype == 0:                                            # 0: no transform. in-degree.
            return Dict
        elif self.po_transtype == 1:                                          # 1: scale in-degree vector at every timepoints with mean 0, st.dev 1. (indeg - mean) / stdev
            std = np.std(list(Dict.values()))
            mean = np.mean(list(Dict.values()))
            return {x:(Dict[x]-mean)/std for x in Dict}
        elif self.po_transtype == 2:                                          # 2: scale in-degree vector at every timepoints with adding 1 and make st.dev 1. (indeg + 1) / stdev
            std = np.std(list(Dict.values()))
            return {x:(Dict[x]+1)/std for x in Dict}
        elif self.po_transtype == 3:                                          # 3: scale in-degree vector at every timepoints with st.dev 1. indeg / stdev
            std = np.std(list(Dict.values()))
            return {x:Dict[x]/std for x in Dict}        
        else:
            raise NameError('Undefined po_transtype.')

    def popularity_t(self, t):
        if self.po_type == "indegree":
            poDict_t = self.data[t]['indegDict']
        elif self.po_type == "log_indegree":
            indDict = self.data[t]['indegDict']
            poDict_t = {x:np.log(1+indDict[x]) for x in indDict}
        elif self.po_type == "closeness_centrality":
            G_t = nx.Graph()
            G_t.add_nodes_from(self.data[0]['nodeList'])
            G_t.add_edges_from(self.data[t]['edgeList'])
            poDict_t = nx.closeness_centrality(G_t)
        elif self.po_type == "betweenness_centrality":
            G_t = nx.Graph()
            G_t.add_nodes_from(self.data[0]['nodeList'])
            G_t.add_edges_from(self.data[t]['edgeList'])
            poDict_t = nx.betweenness_centrality(G_t)
        elif self.po_type == "eigenvector_centrality":
            G_t = nx.Graph()
            G_t.add_nodes_from(self.data[0]['nodeList'])
            G_t.add_edges_from(self.data[t]['edgeList'])
            poDict_t = nx.eigenvector_centrality(G_t)
        elif self.po_type == "pagerank":
            G_t = nx.Graph()
            G_t.add_nodes_from(self.data[0]['nodeList'])
            G_t.add_edges_from(self.data[t]['edgeList'])
            poDict_t = nx.pagerank(G_t, alpha=0.85)
        else:
            raise NameError('Undefined po_type.')
        po_t_all = self.po_transformation(poDict_t)
        return {x:po_t_all[x] for x in self.nodes}

    def popularity_(self):              # length T-1 popularity's list: po=[po[0], ..., po[T-2]]
        po = list()
        for t in range(self.T-1):
            po.append(self.popularity_t(t))
        return po

    def loglik_jt_ars(self, fi_j, beta, j, t):                # log likelihood function for ARS
        beta = np.asarray(beta)
        vv = beta[0]*1 + beta[1]*fi_j + beta[2]*self.po[t-1][j]
        if vv < self.ofBdry :   value = self.y[self.y_idx(t)][j] * vv - self.n[self.n_idx(t)][j] * np.log(1 + np.exp(vv))
        else :                  value = self.y[self.y_idx(t)][j] * vv - self.n[self.n_idx(t)][j] * vv
        return value

    def loglik_j_ars(self, fi_j, beta, j):                    # log likelihood function for ARS
        total = 0
        for t in range(1,self.T):
            total += self.loglik_jt_ars(fi_j, beta, j, t)
        return total

    def loglik_prime_jt_ars(self, fi_j, beta, j, t):            # derivative with respect to fi_j
        beta = np.asarray(beta)
        vv = beta[0]*1 + beta[1]*fi_j + beta[2]*self.po[t-1][j]
        if vv < self.ofBdry :   theta = 1 - 1 / (1 + np.exp(vv))
        else :                  theta = 1
        value = beta[1] * ( self.y[self.y_idx(t)][j] - self.n[self.n_idx(t)][j] * theta )
        return value

    def loglik_prime_j_ars(self, fi_j, beta, j):
        total = 0
        for t in range(1,self.T):
            total += self.loglik_prime_jt_ars(fi_j, beta, j, t)
        return total

    def fi_pri_logdist(self, x):
        if self.fi_dist == "normal": 
            return -0.5 * x**2
        elif self.fi_dist == "log-exponential": 
            ww = -(np.pi/np.sqrt(6) * x + np.euler_gamma)
            return ww - np.exp(ww)
        elif self.fi_dist == "pareto":
            return -(self.pareto_alpha+1) * np.log(x)
        elif self.fi_dist == "gamma":
            theta = 1 / np.sqrt(self.gamma_k)
            return (self.gamma_k-1) * np.log(x) - x / theta    

    def fi_pri_logdist_der(self, x):
        if self.fi_dist == "normal": 
            return -x
        elif self.fi_dist == "log-exponential": 
            ww = -(np.pi/np.sqrt(6) * x + np.euler_gamma)
            return np.pi/np.sqrt(6) * (-1 + np.exp(ww))
        elif self.fi_dist == "pareto":
            return -(self.pareto_alpha+1) / x
        elif self.fi_dist == "gamma":
            theta = 1 / np.sqrt(self.gamma_k)
            return (self.gamma_k-1) / x - 1 / theta  

    def fi_pos_logdist(self, fi_j, beta, j):
        return self.loglik_j_ars(fi_j, beta, j) + self.fi_pri_logdist(fi_j)

    def fi_pos_logdist_der(self, fi_j, beta, j):
        return self.loglik_prime_j_ars(fi_j, beta, j) + self.fi_pri_logdist_der(fi_j)

    def loglik_der_jt_ars(self, fi_j, beta, j, t):            # derivative with respect to beta
        beta = np.asarray(beta)
        xx = np.array([1, fi_j, self.po[t-1][j]])
        vv = beta[0]*1 + beta[1]*fi_j + beta[2]*self.po[t-1][j]
        if vv < self.ofBdry :   theta = 1 - 1 / (1 + np.exp(vv))
        else :                  theta = 1
        value = (self.y[self.y_idx(t)][j] - self.n[self.n_idx(t)][j] * theta) * xx
        return value

    def loglik_der_j_ars(self, fi_j, beta, j):
        total = np.zeros(self.n_pa)
        for t in range(1,self.T):
            total += self.loglik_der_jt_ars(fi_j, beta, j, t)
        return total

    def loglik_hess_jt_ars(self, fi_j, beta, j, t):            # hessian with respect to beta
        beta = np.asarray(beta)
        xx = np.array([1, fi_j, self.po[t-1][j]])
        vv = beta.dot(xx)
        if vv < self.ofBdry :  theta = 1 - 1 / (1 + np.exp(vv))
        else :                 theta = 1
        value = - ( self.n[self.n_idx(t)][j] * theta * (1-theta) ) * np.outer(xx, xx)
        return value

    def loglik_hess_j_ars(self, fi_j, beta, j):
        total = np.zeros((self.n_pa,self.n_pa))
        for t in range(1,self.T):
            total += self.loglik_hess_jt_ars(fi_j, beta, j, t)
        return total

    def Q_j(self, beta_prev, j):                               # Input: beta^(s), j     # Output: Function Q_j(beta|beta^(s))
        def ftn(beta):
            value = 0
            for k in range(self.m):
                value += self.loglik_j_ars(self.z[j][k], beta, j) / self.m
            return value
        return ftn

    def Q_der_j(self, beta_prev, j):                           # Input: beta^(s), j     # Output: D Q_j(beta|beta^(s))
        def ftn(beta):
            value = 0
            for k in range(self.m):
                value += self.loglik_der_j_ars(self.z[j][k], beta, j) / self.m
            return value
        return ftn    

    def Q_hess_j(self, beta_prev, j):                           # Input: beta^(s), j     # Output: D^2 Q_j(beta|beta^(s))
        def ftn(beta):
            value = 0
            for k in range(self.m):
                value += self.loglik_hess_j_ars(self.z[j][k], beta, j) / self.m
            return value
        return ftn    

    def Q_(self, beta_prev):                                   # Input: beta^(s)        # Output: Function Q(beta|beta^(s))
        def ftn(beta):
            value = 0
            for j in self.nodes:
                value += self.Q_j(beta_prev, j)(beta)
            return value
        return ftn

    def Q_der(self, beta_prev):                                # Input: beta^(s)        # Output: Function D Q(beta|beta^(s))
        def ftn(beta):
            value = 0
            for j in self.nodes:
                value += self.Q_der_j(beta_prev, j)(beta)
            return value
        return ftn

    def Q_hess(self, beta_prev):                               # Input: beta^(s)        # Output: Function D^2 Q(beta|beta^(s))
        def ftn(beta):
            value = 0
            for j in self.nodes:
                value += self.Q_hess_j(beta_prev, j)(beta)
            return value
        return ftn
    
    def ars_params(self):
        if self.fi_dist == "normal":
            return -10000, 10000, (float("-inf"), float("inf"))
        elif self.fi_dist == "log-exponential":
            return -3, 10, (float("-inf"), float("inf"))
        elif self.fi_dist == "pareto":       
            c = (self.pareto_alpha-1)*np.sqrt((self.pareto_alpha-2)/self.pareto_alpha)
            return c, c+3, (c, float("inf"))
        elif self.fi_dist == "gamma":
            return 1e-10, 10, (0.0, float("inf"))
        else:
            raise NameError('Undefined (prior) node fitness distribution. Check fi_dist.')

    def ars_(self, beta, j):
        samples = adaptive_rejection_sampling(logpdf=lambda x: self.fi_pos_logdist(fi_j=x, beta=beta, j=j), 
                                              a=self.ars_a, b=self.ars_b, domain=self.ars_domain, n_samples=self.m)
        return samples

    def Estep_(self, beta_prev, max_it=100):                                     # Input: beta_prev      # Output: functions Q(beta|beta_prev), DQ, D^2Q
        for j in self.nodes:
            self.z[j] = self.ars_(beta_prev, j)                          # Draw m samples from posterior fitness distribution, and save it in z
        return self.Q_(beta_prev), self.Q_der(beta_prev), self.Q_hess(beta_prev)     

    def Mstep_(self, beta_prev, fun, jac, hess, max_it=1000):        # Input: beta_prev, E-step Q functions and its derivative, hessian   # Output: beta which maximize Q(beta)
        it = 0
        while it < max_it :
            if it == 0 : bbb = np.zeros(self.n_pa)
            else : bbb = np.random.rand(self.n_pa)-0.5
            res = minimize(fun=self.mf(fun), x0=beta_prev+bbb, method='Newton-CG', jac=self.mf(jac), hess=self.mf(hess))
            succ = res.success
            it += 1
            if succ == True : break
            if it == max_it : print("NotConvergedMstep_", end=' ')
        beta = res.x
        return beta

    def EM_(self, max_it=200, tol=0.001, num=10, disp=False):                              # Input: beta0,  Output: estimated beta
        beta_prev = self.beta0
        it = 0
        startTime = time.time()
        while it < max_it :
            self.beta_list.append(beta_prev)
            if disp == True : print(it, beta_prev)
            it += 1
            Q, Qder, Qhess = self.Estep_(beta_prev)
            beta = self.Mstep_(beta_prev, Q, Qder, Qhess)
            if norm( beta - np.mean(np.array(self.beta_list[-num:]), axis=0) ) < tol:
                if disp == True : print("EM algorithm converged")
                break
            else:
                beta_prev = deepcopy(beta)
            if it == max_it : print("EM algorithm not converged.")
        endTime = time.time()
        if disp == True : print("Running time : %.1f seconds." %(endTime - startTime))
        self.beta_list.append(beta)
        self.beta = deepcopy(beta)
        self.betaCov = self.beta_cov(beta)

    def infoMatrix(self, beta):                                # Input: estimated beta,    # Output: estimated observed information matrix
        part1 = - self.Q_hess(beta)(beta)
        part2 = 0
        for j in self.nodes:
            for k in range(self.m):
                DQ_jk = self.loglik_der_j_ars(self.z[j][k], beta, j)
                part2 += - np.outer(DQ_jk, DQ_jk) / self.m
            DQ_j = self.Q_der_j(beta, j)(beta)
            part2 += np.outer(DQ_j, DQ_j)
        return part1 + part2			

    def beta_cov(self, beta):                                  # Input: estimated beta,    # Output: estimate of the asymptotic covariance matrix of beta
        return inv(self.infoMatrix(beta))

    def printNodesInfo(self, nodeList):
        for j in nodeList:
            print("Node %d, estimated fitness: %.4f" %(j, np.mean(self.z[j])))
            print("Time series indegrees for node %d: " %j, end=' ')
            for t in range(self.T): 
                print(self.data[t]['indegDict'][j], end=' ')
            print("")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    # Load
    f = open('fbwall_GraphData_stack_all.pkl', 'rb')
    Data = pickle.load(f)
    f.close()
    # Apply fpdnm
    fp = FPDNM(data=Data, datatype=2, sampling_method="initial", n_eff=300, beta0=[-5.0000, 1.0000, 0.0000], M=1000)
    fp.EM_(max_it=150, tol=0.00001, disp=True)
    print("Estimated beta parameter: ", fp.beta)
    print("Estimated standard error: ", np.sqrt(np.diag(fp.betaCov)))
    print("Estimated fitness: ", {x:str(np.round(np.mean(fp.z[x]),3)) for x in fp.z})
    print("Mean and St.dev of estimated fitness: %.4f, %.4f." %(np.mean([np.mean(x) for x in fp.z.values()]), np.std([np.mean(x) for x in fp.z.values()])))
    fig = plt.figure(figsize=(20,5))
    plt.plot( [np.mean(x) for x in fp.z.values()])
    plt.title('Estimated Fitness (%d nodes)' %len(fp.nodes))
    plt.show()
    # Save Graph Data
    f = open('fbwall_GraphData_stack_all_result.pkl', 'wb')
    pickle.dump(fp, f, protocol=0)
    f.close()









