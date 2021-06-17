#/usr/bin/python3 
import math
import sys
import numpy as np
import scipy as scipy
from scipy import sparse
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time 

######################################################################################################################################################
# Pentru mai multe detalii, algoritmul este bine descris in https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/primal-dual.pdf
######################################################################################################################################################

# Load time series data: S&P 500 price log.
dataset = np.loadtxt(open('snp500.txt', 'rb'), delimiter=",", skiprows=1)  # y
datasetLength = dataset.size


D = np.zeros( ( datasetLength - 2, datasetLength ) )
for i in range( datasetLength - 2) :
    D[ i ][ i ] = 1 
    D[ i ][ i + 1 ] = -2 
    D[ i ][ i + 2 ] = 1 
    

def l1_trend( landa, y, D ) :
    
    #Parametrii initiali
    MAX_ITER  = 40
    MAX_LITER = 20
    #Lungimea datelor
    N = len( y )
    #Lungimea x
    M = N - 2
    #Parametrii backtracking, vezi prin ppt-ul de mai sus
    MU = 2
    ALPHA = 0.01
    BETA = 0.5
    STEP = 1
    
    T = 1e-10
    TOL = 1e-4
     
    def prim_problem( v, x, D ) :
        return 0.5 * np.transpose( v ) @ D @ np.transpose( D ) @ v + landa * np.sum( np.abs( D @ x ) )
    
    # 0.5 vT DDT v − yT DT v
    def dual_problem( v, y, D ) :
        return 0.5 * np.transpose( v ) @ D @ np.transpose( D ) @ v - np.transpose( y ) @ np.transpose( D ) @ v
    
    
    #Initializari
    z = np.zeros( M )
    mu1 = np.ones( M )
    mu2 = np.ones( M )

    #Constrangeri
    f1 = z - landa  
    f2 = -z - landa

    #Precalculeaza matricea D * D.T
    DDT = D @ np.transpose( D )
    
    for iterations in range( MAX_ITER ) :
        
        print( "Sunt la iteratia : " + str(iterations) )

        #evalueaza problema primala in v
        prim_val = prim_problem( z, y - (np.transpose(D) @ z), D )
        
        #evalueaza problema duala in v
        dual_val = -dual_problem( z, y, D )    
        gap = prim_val - dual_val 
        
        #daca diferenta dintre solutia in problema primala si cea duala e aprope zero, am gasit minim 
        #asta se intampla pentru ca strong duality
        print("gap : " + str(gap) )
        print( "step : " + str(STEP) )
        if gap <= TOL :
            return y - np.transpose( D ) @ z
        
         
        if STEP >= 0.2 :   
            T = max( 2 * M * MU / gap, 1.2 * T ) 
        

        rz = DDT @ z - D @ y + mu1 - mu2

        #scoatele inafara pentru optimizare la inmultire de matrici diagonale
        aux_1 = np.multiply(1/f1,mu1)
        aux_2 = np.multiply(1/f2,mu2)
        
        #facem matricea diagonala abia dupa. Aici era o inmutire de 2 mat diag, dar asta e echivalent
        j1 = np.diag( aux_1 )
        j2 = np.diag( aux_2 ) 


        S = DDT - np.diag( np.multiply( aux_1, aux_2) )
        dbg_start = time.time();

        r = D @ y - DDT @ z + (1/T) / f1 - (1/T) / f2 

    
        dz = sparse.linalg.spsolve( sparse.csr_matrix(S), r)
                    
        dmu1 = -( mu1 + (1/T) / f1 + j1 @ dz )
        dmu2 = -( mu2 + (1/T) / f2 - j2 @ dz )        
                
        resDual = rz
        resCent = np.concatenate( (-1/T - np.multiply(mu1,f1), -1/T - np.multiply(mu2,f2)), axis=0 )
        residual= np.concatenate( (resDual, resCent), axis=0 )       
        
        
        stap = 1
        for liter in range( MAX_LITER ) :
            #incearca sa te deplasezi
            newz    =  z  + STEP*dz
            newmu1  =  mu1 + STEP*dmu1
            newmu2  =  mu2 + STEP*dmu2
            newf1   =  newz - landa
            newf2   = -newz - landa

            #calculeaza noua pozitie
            newResDual = DDT @ newz - D @ y + newmu1 - newmu2
            newResCent = np.concatenate( (-1/T - np.multiply(newmu1,newf1), -1/T - np.multiply(newmu2,newf2)), axis=0 )
            newResidual= np.concatenate( (newResDual, newResCent), axis=0 )
            
            #daca suntem pe descrestere oprestete
            if max( max( newf1 ), max(newf2) ) < 0 and np.linalg.norm( newResidual ) <= ( 1-ALPHA*STEP ) * np.linalg.norm( residual ) :
                break
            
            #micsoreaza pasul
            STEP = BETA * STEP  
        
        #pregateste valorile pentru urmatoarea iteratie
        z = newz
        mu1 = newmu1
        mu2 = newmu2
        f1 = newf1
        f2 = newf2
        
    return y - np.transpose( D ) @ z


x = l1_trend( 100, dataset, D )
plt.plot( np.linspace( 1, len(dataset), len(dataset) ), dataset )
plt.plot( np.linspace( 1, len(x), len(x) ), x )
plt.show()
    