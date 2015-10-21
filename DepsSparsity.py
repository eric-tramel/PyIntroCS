import matplotlib.pyplot as plt     # PyPlot for Plotting
import numpy as np
import copy

def ShowSignal(x):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    f = plt.figure(figsize=(15,5))
    plt.plot(x);
    plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Index", fontsize=18)
    plt.ylabel("Value", fontsize=18)

def CompareER(xname,x,yname,y,TestedK):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    f = plt.figure(figsize=(12,5))
    
    hx = plt.plot(TestedK,x,label=xname,linewidth=2);
    hy = plt.plot(TestedK,y,label=yname,linewidth=2);
    plt.ylim((0.0,1.0))

    plt.axis("tight")
    plt.grid(True)
    plt.xlabel("$K$ Retained", fontsize=18)
    plt.ylabel("Energy Retained", fontsize=18)  
    plt.legend()


def ShowDecay(x,scalemode='linear'):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    n = np.size(x)
    x = np.reshape(x,(n,1))

    xsort = sorted(abs(x),reverse=True)
    
    f = plt.figure(figsize=(12,5))
    plt.yscale('log')
    # plt.yscale(scalemode)
    # plt.xscale(scalemode)
    plt.ylim((1e-10,1e1))
    plt.xlim ((0,n))
    hx = plt.plot(xsort,linewidth=2);
    
    # plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Sorted Index", fontsize=18)
    plt.ylabel("Magnitude $|\\tilde{x}_i|$", fontsize=18)  

def ShowWaveletDecay(w):
    x = np.reshape(w[0],(np.size(w[0]),1))

    for l in range(1,np.size(w)):
        h = np.reshape(w[l][0],(np.size(w[l][0]),1))
        v = np.reshape(w[l][1],(np.size(w[l][1]),1))
        d = np.reshape(w[l][2],(np.size(w[l][2]),1))
        x = np.concatenate((x,h,v,d),axis=0)

    ShowDecay(x)

def CompareHistograms(xname,x,yname,y):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')    
    
    f = plt.figure(figsize=(12,5))

    plt.subplot(121)
    hx = plt.hist(abs(x),bins=50,label=xname);
    plt.ylim((0,1000))
    plt.xlim((0,4))
    plt.title("$K$-Sparse",fontsize=20)
    plt.grid(True)
    plt.ylabel("Frequency", fontsize=18)
    plt.xlabel("Magnitude $|x_i|$", fontsize=18)  

    plt.subplot(122)
    hy = plt.hist(abs(y),bins=50,label=yname);
    plt.ylim((0,1000))
    plt.xlim((0,4))
    plt.title("Laplacian",fontsize=20)
    plt.grid(True)
    plt.ylabel("Frequency", fontsize=18)
    plt.xlabel("Magnitude $|x_i|$", fontsize=18)  

def CompareDecayTwo(xname,x,yname,y,scalemode='linear'):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')

    n = np.size(x)
    
    xsort = sorted(abs(x),reverse=True)
    ysort = sorted(abs(y),reverse=True)
    
    f = plt.figure(figsize=(12,5))
    
    hx = plt.plot(xsort,label=xname,linewidth=2);
    hy = plt.plot(ysort,label=yname,linewidth=2);
    plt.yscale('log')
    # plt.yscale(scalemode)
    # plt.xscale(scalemode)
    plt.ylim((1e-10,1e1))
    plt.xlim((0,n))
    # plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Sorted Index", fontsize=18)
    plt.ylabel("Magnitude $|\\tilde{x}_i|$", fontsize=18)  
    plt.legend()

def CompareDecayThree(xname,x,yname,y,zname,z):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    xsort = sorted(abs(x),reverse=True)
    ysort = sorted(abs(y),reverse=True)
    zsort = sorted(abs(z),reverse=True)
    
    f = plt.figure(figsize=(12,5))
    
    hx = plt.plot(xsort,label=xname,linewidth=2);
    hy = plt.plot(ysort,label=yname,linewidth=2);
    hz = plt.plot(zsort,label=zname,linewidth=2);    
    
    plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Index", fontsize=18)
    plt.ylabel("Value", fontsize=18)  
    plt.legend()
    
def CompareEnergyCompaction(xname,x,yname,y,zname,z):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    f = plt.figure(figsize=(10,5))
    
    hx = plt.plot(x,label=xname,linewidth=2);
    hy = plt.plot(y,label=yname,linewidth=2);
    hz = plt.plot(z,label=zname,linewidth=2);    
    
    plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Coefficients Retained", fontsize=16)
    plt.ylabel("Energy Retained (\%)", fontsize=16)  
    plt.legend()
    
def CompareMSE(xname,x,yname,y,zname,z):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    f = plt.figure(figsize=(10,5))
    
    hx = plt.plot(x,label=xname,linewidth=2);
    hy = plt.plot(y,label=yname,linewidth=2);
    hz = plt.plot(z,label=zname,linewidth=2);    
    
    plt.axis("tight")
    plt.grid(True)
    plt.yscale('log')
    plt.ylim((1e-10,1e1))
    plt.xlabel("Coefficients Retained", fontsize=16)
    plt.ylabel("$1/N||\mathbf{x}-\mathbf{x}_K||_2^2$", fontsize=16)  
    plt.legend()

def ShowWaveletCoeffs(W,SuppressBaseband=False,Magnitude=True,cmapname='jet'):
    L = np.size(W)
    bb = np.array(W[0])
    if SuppressBaseband:
        bb = np.nan * bb
    X = np.concatenate((bb,W[1][0]),axis=1)
    tmp = np.concatenate((W[1][1],W[1][2]),axis=1)
    X = np.concatenate((X,tmp),axis=0)

    for l in range(2,L):
        X = np.concatenate((X,W[l][0]),axis=1)
        tmp = np.concatenate((W[l][1],W[l][2]),axis=1)
        X = np.concatenate((X,tmp),axis=0)

    f = plt.figure(figsize=(20,20),dpi=300)
    if Magnitude:
        plt.matshow(np.abs(X),cmap=cmapname);
    else:
        plt.matshow(X,cmap=cmapname);

def ShowWaveletDecay(w):
    ShowDecay(WaveletToVector(w))


def WaveletToVector(w):    
    x = np.reshape(w[0],(np.size(w[0]),1))

    for l in range(1,np.size(w)):
        h = np.reshape(w[l][0],(np.size(w[l][0]),1))
        v = np.reshape(w[l][1],(np.size(w[l][1]),1))
        d = np.reshape(w[l][2],(np.size(w[l][2]),1))
        x = np.concatenate((x,h,v,d),axis=0)
    return x

def WaveletTopKApproximation(wim,K):
    W = copy.deepcopy(wim)

    l = np.size(W)
    x = WaveletToVector(W)
    x = sorted(abs(x),reverse=True)
    ## Get the threshold
    T = abs(x[K])

    W = list(W)
    W[0][abs(W[0]) < T] = 0.0

    for level in range(1,l):  
        W[level] = list(W[level])
        for dir in [0,1,2]:
            thisw = W[level][dir]
            thisw[abs(thisw) < T] = 0.0
            W[level][dir] = thisw

    return W





