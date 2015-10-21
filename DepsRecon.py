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

def KSparseApproximation(x,K):
    """ Retain the top K coefficients in  x.
    
    Given a signal, find the top K-magnitude coefficients.
    Return a signal which retains these identified coefficients
    and sets the rest to 0.
    
    Input
    ---
    x: array_like, float
       The signal that we wish to approximate
       
    K: int
       Number of coefficients to retain.
    
    Returns
    ---
    xT: array_like, float
        The K-Sparse approximation of x
    S:  array_like, int
        An array of the support index locations in x.    
    """
    S = np.argsort(abs(x),axis=0)[::-1][:K]
    xT = np.zeros_like(x)
    xT[S] = x[S]
    
    return xT, S

def RandomLaplaceSignal(scale,N):
    """ Random realization of an iid Laplace signal. 
    
        Input
        ---
        K: float
           Scale parameter for the Laplacian distribution.
        N: int
           Signal dimensionality (length)
        
        Returns
        ---
        x: array_like, float
           The iid Laplace signal realization, shaped ``(n,1)``                
    """
    x = np.random.laplace(0.0,scale=scale,size=(N,1))    
    return x

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
    plt.plot(y,'-xr');
    plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Index", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    
    mseval = MeanSquareError(x,y)

def MeanSquareError(x,y):
    mseval = np.mean(np.power(x-y,2))    
    return mseval

def ShowHistory(mseval,resid):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    f = plt.figure(figsize=(10,5),dpi=300)
    plt.plot(mseval,'-b',linewidth=2,label="$\\frac{1}{N}||x - x^{(t)}||_2^2$");
    plt.plot(resid,'-r',linewidth=2,label="$\\frac{1}{M}||y - F x^{(t)}||_2^2$");
    plt.yscale('log')
    plt.ylim((1e-10,1e1));
    plt.xlim((0,np.size(mseval)-1))
    plt.grid(True)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    plt.title("Evolution of Reconstruction")
    plt.legend()
