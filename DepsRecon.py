import matplotlib.pyplot as plt     # PyPlot for Plotting
import numpy as np

def RandomKSparseSignal(K,N):
    """ Random realization of a K-sparse signal.
    
        Assumes that all non-zero components of the realization
        are drawn iid from a zero-mean Gaussian with a variance 
        of ``1``. 
    
        Input
        ---
        K: int
           Number of non-zeros in the signal.
        N: int
           Signal dimensionality (length)
        
        Returns
        ---
        x: array_like, float
           The K-sparse signal realization, shaped ``(n,1)``
        s: array_like, bool
           The support of ``x``, shaped ``(n,1)``. Has value
           ``True`` at every non-zero location of ``x`` and
           ``False`` everywhere else.
           
        
    """
    x = np.random.randn(N,1)                # Generate iid Gaussian samples     
    s = np.zeros((N,1),dtype=bool)          # Cast ``0.0`` as ``False``
    si = np.random.permutation(N)[1:K]      # Random support locations
    s[si] = True                            # Flag on-support values
    x[np.invert(s)] = 0.0                   # Suppress off-support values of ``x``
    
    return x, s

def ShowSignal(x):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    f = plt.figure(figsize=(15,5))
    plt.plot(x);
    plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Index", fontsize=18)
    plt.ylabel("Value", fontsize=18)

def ShowMeasurements(y):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    n = np.size(y)

    f = plt.figure(figsize=(15,5))
    plt.bar(np.arange(0,n),y);
    plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Index", fontsize=18)
    plt.ylabel("Value", fontsize=18)

def ShowProjection(F):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    plt.matshow(F,cmap='jet');
    plt.axis("tight")
    plt.xlabel("N", fontsize=18)
    plt.ylabel("M", fontsize=18)

def ShowRecovery(x,y):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    f = plt.figure(figsize=(10,5),dpi=300)
    plt.plot(x,':ob');
    plt.plot(y,':xr');
    plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Index", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    
    mseval = MSE(x,y)

def MSE(x,y):
    mseval = np.mean(np.power(x-y,2))    
    return mseval