import matplotlib.pyplot as plt     # PyPlot for Plotting

def ShowSignal(x):
    plt.rc('text',usetex=True)
    plt.rc('font', family='serif')
    
    f = plt.figure(figsize=(15,5))
    plt.plot(x);
    plt.axis("tight")
    plt.grid(True)
    plt.xlabel("Index", fontsize=18)
    plt.ylabel("Value", fontsize=18)
    
def CompareDecay(xname,x,yname,y,zname,z):
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