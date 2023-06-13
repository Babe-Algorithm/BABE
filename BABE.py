#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 17:47:53 2023

@author: rutabinkyte
"""
import pandas as pd
import numpy as np



#### BABE #######

def get_priors(E_G, Z_G ,Z_EG,P_G, Eval, Gval, Zval,d):
    primes = []
    
    k=0
 
    while True:
        k+=1
        E_G_1 = update(E_G, Z_G,Z_EG,P_G, Eval, Gval,Zval,d)#updating for the next step
        E_G = E_G_1
        primes.append(E_G)
    
        if len(primes) > 2:
            diff=abs(pd.Series(primes[-1].values())-pd.Series(primes[-2].values()))
            if all(i <0.0001 for i in diff): #stopping condition
     
                break
    print("Number of iterations: ", k)
            
    return primes[-1] 

def update(E_G, Z_G,Z_EG, P_G,Eval, Gval,Zval,d):
    names = ['E_d', 'G', 'Z_d']
    E_G_new = E_given_G_Uniform( Eval, Gval)
    
    for EG in E_G.keys(): #over all E|G
          
        z_args = []
        for z in range(Zval): #over all Z values 

            d_args = []
            for e in range(Eval): #over all E values , where G is fixed
                g = EG[1] #G is fixed

                z_eg = Z_EG.get((e,g,z)) #and z is fixed

                d_arg = z_eg*E_G.get((e,g))

                d_args.append(d_arg)
            denominator = sum(d_args)# sum over e P(Z|EG)P(E|G)
            
            e=EG[0]
            g=EG[1]
            
            numerator = Z_G.get((g,z))*Z_EG.get((e,g,z))*E_G.get(EG) #P(Z|G)*P(Z|EG)*P(E|G)
          
            if denominator == 0:
                denominator=0.000000001
            
            fraction = numerator/denominator #inference for one value of S
               
            z_args.append(fraction)
        e_given_g = sum(z_args) #sum over z
        E_G_new[EG]=round(e_given_g,4)

           
    return E_G_new


    
#### INFERENCE ####

#names = ['E_d', 'G', 'Z_d']

def inferE(df, names, E_given_G, Z_G, Z_EG, nE, d): #Find P(M|A,S) Bayesian Network inference
    E_probs = [] #probability of E value 
    E_values = []
    

    for i in range(df.shape[0]):
   
       
        z=df.iloc[i].loc[names[2]] #value of Z (in data)

        g=df.iloc[i].loc['G'] #the group

        P_z_g = Z_G.get((g,z))
        if P_z_g ==0:
            P_z_g=0.000001
        


        e_probs = [] #probabilities of all E values, to pick the highest
        
 
        
        for e in range(nE):

            

            P_z_e_g = Z_EG.get((e,g,z))
            if P_z_e_g ==0:
                P_z_e_g=0.000001
                
          
            P_eg =  E_given_G.get((e,g)) 
            if P_eg==0:
                P_eg=0.000001
           

            

            P_e_gz=round(P_z_e_g* P_eg/P_z_g, 8)
      
            e_probs.append(P_e_gz)



        #normalize
        Sum = sum(e_probs)

        for k in range(len(e_probs)):
            e_probs[k] = round(e_probs[k]/Sum,d)

        prob = max(e_probs)

        E_probs.append(e_probs)
        value = e_probs.index(prob)
        E_values.append(value)

    E_probs = pd.DataFrame(E_probs)
    return  E_probs, E_values


def inferE_fast(df, names, E_given_G, Z_G, Z_EG, nE,nZ, d): #Find P(M|A,S) Bayesian Network inference
    E_probs = {} #probability of E value 
    E_values = {}
    
    for g in range(2):
        print('g',g)
    
    
        for z in range(nZ):

            print("z: ", z)
        
            P_z_g = Z_G.get((g,z))
            if P_z_g ==0:
                P_z_g=0.000001
            print("P_z_g", P_z_g)
            

            e_probs = [] #probabilities of all E values, to pick the highest

            for e in range(nE):
                print('e', e)


                P_z_e_g = Z_EG.get((e,g,z))
                if P_z_e_g ==0:
                    P_z_e_g=0.000001
                print('P_z_e_g',P_z_e_g)

                P_eg =  E_given_G.get((e,g)) 
                print('P_eg',P_eg)


                P_e_gz=round((P_z_e_g* P_eg)/P_z_g, d) #Alternative
                print('P_e_gz', P_e_gz)
                e_probs.append(P_e_gz)


            Sum = sum(e_probs)
            for k in range(len(e_probs)):
                e_probs[k] = round(e_probs[k]/Sum,d)

            prob = max(e_probs)
     
            E_probs[(g,z)]= e_probs 
            value = e_probs.index(prob)
            E_values[(g,z)]=value


    return  E_probs, E_values


#####Fairness Statistics####



def SP(colname,  df):
    if (df.loc[(df[colname]==1)&(df['G']==0)].shape[0]==0) & (df.loc[(df['G']==0)].shape[0]==0):
        g0=1
    else:
    
        g0=df.loc[(df[colname]==1)&(df['G']==0)].shape[0]/df.loc[(df['G']==0)].shape[0]
    
    if (df.loc[(df[colname]==1)&(df['G']==1)].shape[0]==0) & (df.loc[(df['G']==1)].shape[0]==0):
        g1=1
    else:
        g1 = df.loc[(df[colname]==1)&(df['G']==1)].shape[0]/df.loc[(df['G']==1)].shape[0]
    
    sp = g0-g1
    
    return sp 


####Utils Parameters#####


def manhattan_dist(a,b):
    '''
    
    manhattan distance between two vectors
    Parameters
    ----------
    a : vector
        vector with values 1
    b : vector
        vector with values 2

    Returns
    -------
    Int
        The distance between two vectors

    '''
    
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))
    

#names = ['E_d', 'G', 'Z_d']
def Z_given_EG(df,names, nE, nG, nZ):
    '''
    

    Parameters
    ----------
    df : data frame
        the data from which computing frequencies
    names : list of strings
        The names of variables in the data frame
    nE : int
        number of possible values for E -explanatory variable
    nG : TYPE
        number of possible values for G - group variable
    nZ : TYPE
        number of possible values for Z - biased score

    Returns
    -------
    dict : dictionary
        CPT of Z|E,G

    '''
    

    #E|G|Z
    E=[i for i in range(nE)]
    G=[i for i in range(nG)]
    Z=[i for i in range(nZ)]
 
    keys=[]

    #Create keys list
    for e in E:
        for g in G:
            for z in Z:
                key = (e,g,z)
                keys.append(key)
                
    l = []

    for key in keys:
        
        if df[(df[names[0]] ==key[0])&(df[names[1]] == key[1])].shape[0] !=0:
   
    
               
            p = df[(df[names[0]] ==key[0])&(df[names[1]] ==key[1])&(df[names[2]] ==key[2])].shape[0]/df[(df[names[0]] ==key[0])&(df[names[1]] == key[1])].shape[0]
            l.append(p)
        else:
            p=0.000001
            l.append(p)
    dict = {key:np.nan for key in keys}
    for i in range(len(dict.keys())):
        #if key[0] ==0
        dict[list(dict.keys())[i]] = l[i]
        
    return dict

#names = ['E_d', 'G', 'Z_d']

def Z_given_G(df,names, nG, nZ):
    '''
    

   Parameters
    ----------
    df : data frame
        the data from which computing frequencies
    names : list of strings
        The names of variables in the data frame

    nG : TYPE
        number of possible values for G - group variable
    nZ : TYPE
        number of possible values for Z - biased score

    Returns
    -------
    dict : dictionary
        CPT of Z|G

    '''
    G=[i for i in range(nG)]
    Z=[i for i in range(nZ)]
    keys=[]

    #Create keys list
    for g in G:
        for z in Z:
            key = (g,z)
            keys.append(key)
                
    l = []

    for key in keys:
        if df[(df[names[1]] ==key[0])].shape[0] !=0:
            p = df[(df[names[1]] ==key[0])&(df[names[2]] ==key[1])].shape[0]/df[(df[names[1]] ==key[0])].shape[0]
            l.append(p)
        else:
            p=0.000001 #avoid zero devision
            l.append(p)
    dict = {key:np.nan for key in keys}
    for i in range(len(dict.keys())):
  
        dict[list(dict.keys())[i]] = l[i]
    return dict
    

#names = ['E_d', 'G', 'Z_d']

def E_given_G(df,names, nE, nG):
    '''
    The true learnable variables to check if present in data 

    Parameters
    ----------
    df : data frame
        the data from which computing frequencies
    names : list of strings
        The names of variables in the data frame

    nG : TYPE
        number of possible values for G - group variable
    nE : int
        number of possible values for E -explanatory variable

    Returns
    -------
    dict : dictionary
        CPT of E|G

    '''
    G=[i for i in range(nG)]
    E=[i for i in range(nE)]
    keys=[]
    #Create keys list
    for g in G:
        for e in E:
            key = (e,g)
            keys.append(key)
                
    l = []

    for key in keys:
        if df[(df[names[1]] ==key[1])].shape[0] !=0:
            p = df[(df[names[0]] ==key[0])&(df[names[1]] ==key[1])].shape[0]/df[(df[names[1]] ==key[1])].shape[0]
            l.append(p)
        else:
            p=0.000001
            l.append(p)
            
    dict = {key:np.nan for key in keys}
    for i in range(len(dict.keys())):
      
        dict[list(dict.keys())[i]] = round(l[i],4)
    return dict


def E_given_G_Uniform(nE, nG):
    '''
    The initial uniform distribution of E|G for the get_priors() algorithm

    Parameters
    ----------
    nG : int
        number of possible values for G - group variable
    nE : int
        number of possible values for E -explanatory variable

    Returns
    -------
    dict : dictionary
        Uniform CPT of E|G

    '''
    G=[i for i in range(nG)]
    E=[i for i in range(nE)]
    keys=[]
    #Create keys list
    for g in G:
        for e in E:
            key = (e,g)
            keys.append(key)
                


            
    dict = {key:np.nan for key in keys}
    for i in range(len(dict.keys())):
     
        dict[list(dict.keys())[i]] = round(1/nE,2)
    return dict



def Prob(df,name, n):
    '''
    marginal probability for any variable

    Parameters
    ----------
    df : data frame
        data for calculating frequencies
    name : string
        column name
    n : int
       number of values (domain of the variable)

    Returns
    -------
    l : list
        proababilty distribution

    '''
    
    V=[i for i in range(n)]
                
    l = []
    

    for v in V:
        p = df[(df[name] ==v)].shape[0]/df.shape[0]
        l.append(p)
    return l
  


### Data ####

def generate_data(n, p, bias0, bias1, mean0, mean1, sd0, sd1, threshold, seed):
 
    

    
    np.random.seed(seed)

    #generating G
    G = np.random.binomial(1,p,n)
    #E = [np.random.normal(mean0,sd1) if g ==0 else np.random.normal(mean1,sd2) for g in G]

    #generating E
    E=[]
    for g in G:
        if g ==0:
            e = np.random.normal(mean0,sd0)
            while e<0 or e>99:
                e = np.random.normal(mean0,sd0)
            E.append(e)
        else:
            e=np.random.normal(mean1,sd1)
            while e<0 or e>99:
                e = np.random.normal(mean1,sd1)
            E.append(e)

    #adding bias and generating Z   

    Z=[]

    for e,g in zip (E,G):

       
        if g==0:
            bias = np.random.normal(bias0,0.05)
            if e<50:
                
                z=e+(e*bias)
            else:
                z = e+(99-e)*bias
                

        elif g==1:
            bias = np.random.normal(bias1,0.05)
            if e<50:
                
                z=e+(e*bias)
            else:
                z = e+(99-e)*bias
   
        
        Z.append(z)




    df = pd.DataFrame({'G': G, 'E': E,'Z': Z})
    df['E_d']=df.E.round().astype(int) #discretizing by rounding
    df.loc[df.E_d<0, 'E_d']=0
    df.loc[df.E_d>99, 'E_d']=99

    df['Z_d']=df.Z.round().astype(int) #discretizing by rounding
    df.loc[df.Z_d<0, 'Z_d']=0
    df.loc[df.Z_d>99, 'Z_d']=99

  

    df['YE'] = [1 if x>threshold else 0 for x in df.E_d]
    df['YZ'] = [1 if x>threshold else 0 for x in df.Z_d]


    
    source, estimate, test = np.split(df.sample(frac=1), [int(.4*len(df)), int(.8*len(df))])
    return source, estimate,test
    
    


    
    


