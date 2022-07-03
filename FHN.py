import sys
sys.path.append("/Users/TorOla/Desktop/NTNU/Master Thesis/Python/Classes") # local path; change it to where the file 'CMSDE.py' is stored on your device.
from CMSDE import CMSDE
from CMSDE import generateRandomVariables
import numpy as np
import matplotlib.pyplot as plt


runSimpleDemo = False
testMeanSquareConvergence = False
testErgodicity = True


# Exact solution of nonlinear ODE
def exactStep(x,h,eqParams): 
    eps = eqParams["eps"]
    beta = eqParams["beta"]
    M = eqParams["M"]
    res = np.zeros((M,2),dtype=float)
    e = np.exp(-2*h/eps)
    res[:,0] = x[:,0] / np.sqrt(e + (1-e)*(x[:,0])**2)
    res[:,1] = np.exp(-h)*(x[:,1]-beta)+beta
    return res

# Right-hand-side of nonlinear ODE
def B(x,eqParams): 
    eps = eqParams["eps"]
    beta = eqParams["beta"]
    res = np.zeros(np.shape(x),dtype=float)
    res[:,0] = (1/eps)*(x[:,0]-x[:,0]**3)
    res[:,1] = beta - x[:,1]
    return res

# Jacobian of B
def JB(x,eqParams):
    M = eqParams["M"]
    d = eqParams["d"]
    eps = eqParams["eps"]
    J = np.zeros((M,d,d),dtype=float)
    J[:,0,0] = (1-3*x[:,0]**2)/eps
    J[:,0,1] = 0
    J[:,1,0] = 0
    J[:,1,1] = -1
    return J

# Exact solution of implicit equation
def implicitExactStep(x,h,eqParams):
    eps = eqParams["eps"]
    beta = eqParams["beta"]
    M = eqParams["M"]
    res = np.zeros((M,2),dtype=float)
    a = (eps/h-1)
    b = (eps/h)*x[:,0]
    y = ((-b+np.sqrt(b**2+4*(a/3)**3))/2)**(1/3)
    z = a/(3*y)
    res[:,0] = z-y
    res[:,1] = (x[:,1]+h*beta)/(1+h)
    return res

# Rosenbrock approximation to solution of nonlinear ODE
def rosenbrockStep(x,h,eqParams):
    eps = eqParams["eps"]
    beta = eqParams["beta"]
    a = h/eps
    if a < 1:
        M = eqParams["M"]
        res = np.zeros((M,2),dtype=float)
        res[:,0] = x[:,0] + (a*(x[:,0]-x[:,0]**3))/(1-a*(1-3*x[:,0]**2))
        res[:,1] = x[:,1] + h*(beta-x[:,1])/(1+h)
        return res
    else:
        raise Exception("h/eps must be positive and smaller than 1")
    
# Full drift term 
def F(x,eqParams):
    eps = eqParams["eps"]
    beta = eqParams["beta"]
    gamma = eqParams["gamma"]
    res = np.zeros(np.shape(x),dtype=float)
    res[:,0] = (1/eps)*(x[:,0]-x[:,0]**3-x[:,1])
    res[:,1] = gamma*x[:,0] + beta - x[:,1]
    return res

# Jacobian of F
def JF(x,eqParams):
    M = eqParams["M"]
    d = eqParams["d"]
    eps = eqParams["eps"]
    J = np.zeros((M,d,d),dtype=float)
    J[:,0,0] = (1-3*x[:,0]**2)/eps
    J[:,0,1] = -1/eps
    J[:,1,0] = gamma
    J[:,1,1] = -1
    return J
        
# Exact solution to implicit equation associated with the drift-implicit Euler-Maruyama method. 
def DIEMimplicitExactStep(x,h,eqParams,dW):
    eps = eqParams["eps"]
    beta = eqParams["beta"]
    gamma = eqParams["gamma"]
    M = eqParams["M"]
    res = np.zeros((M,2),dtype=float)
    a = (eps/h)-1+((h*gamma)/(1+h))
    b = eps/h*x[:,0]-((x[:,1]+h*beta+dW[:,1])/(1+h)) + eps/h*dW[:,0]
    y = ((-b+np.sqrt(b**2+4*(a/3)**3))/2)**(1/3)
    z = a/(3*y)
    res[:,0] = z-y
    res[:,1] = (x[:,1]+h*gamma*(z-y)+h*beta+dW[:,1])/(1+h)    
    return res

# Model parameters 
eps = 1.0
gamma = 1/eps
beta = 1.0

# Noise components
sig1 = 0.5
sig2 = 0.5

# Linear part of drift 
A = np.array([[0.0,-1/eps],[gamma,0.0]])
d = np.shape(A)[0]

# Diffusion matrix
sig = np.array([[sig1,0.0],[0.0,sig2]])    

# Logarithmic norm of A
mu = np.max(np.linalg.eigvals(0.5*(A+A.T)))
print(f"Logarithmic norm: {mu}")

# Simulation parameters 
t0 = 0.0 # start time
T = 10.0 # end time
h = 0.01 # step-size 
N = int((T-t0)/h) # Number of discrete time-points 
M = 1000 # Number of simulated Brownian paths

# Initial value 
X0 = np.array([2.0,0.0])

print("Number of sample points:",(T-t0)/h)
print("Stepsize:",h)

# Value of alpha from Assumption 3.3 (see report)
alpha = 0.8 # true alpha 
numAlpha = 1/(2*h)*np.log(1+2*alpha*h) # numerical alpha (in report: \tilde{\alpha}_0 )
print(f"Alpha:             {alpha}")
print(f"Numerical alpha:   {numAlpha}")

# Value of K from Assumption 3.3 (see report) 
K = 0.25*((alpha*eps+1)**2/eps + beta**2/(1-alpha))
    
# Generate random variables Y used in splitting/composition methods and dW used in drift-implicit Euler-Maruyama method. 
Y,dW = generateRandomVariables(A,sig,h,M,N)


# Plotting colors
redColor = np.array([242.0, 17.0, 17.0])/255.0
blueColor = color= np.array([17.0,17.0,242.0])/255.0
lightRedColor = np.array([255.0,140.0,140.0])/255.0
lightBlueColor = np.array([140.0,140.0,255.0])/255.0
greenColor=np.array([15.0,205.0, 53.0])/255
purpleColor=np.array([177.0,13.0,236.0])/255
goldColor = np.array([204.0,209.0,53.0])/255
orangeColor = np.array([238.0,156.0,16.0])/255

# Equation parameters 
eqParams = {"A": A, "sig": sig, "gamma":gamma,"eps":eps,"beta":beta, "M":M,"d":d}

# Input data for our nuumerical methods
data = {"eqParams":eqParams,"X0":X0,"M":M, "h":h, "t0":t0, "T":T,"B":B, "JB":JB,\
        "F":F, "JF":JF,"Y":Y, "dW":dW,\
        "exactStep":exactStep,"implicitExactStepIsDefined":True, "implicitExactStep":implicitExactStep,\
        "DIEMimplicitExactStepIsDefined":True, "DIEMimplicitExactStep":DIEMimplicitExactStep,\
        "rosenbrockStep":rosenbrockStep, "randomInit":False}


# Create 'CMSDE' objects for each numerical method (see file 'CMSDE.py')
print("---------------------------------------------")
print("Creating Lie-Trotter object...")
LT = CMSDE({**data, **{"compoScheme":"LT", "odeStepScheme":"exact"}})

print("---------------------------------------------")
print("Creating Strang object...")
S = CMSDE({**data, **{"compoScheme":"S", "odeStepScheme":"exact"}})

print("---------------------------------------------")
print("Creating Lie-Trotter implicit Euler object...")
LTIE = CMSDE({**data, **{"compoScheme":"LT", "odeStepScheme":"IE"}})

print("---------------------------------------------")
print("Creating Strang implicit Euler object...")
SIE = CMSDE({**data, **{"compoScheme":"S", "odeStepScheme":"IE"}})

print("---------------------------------------------")
print("Creating Lie-Trotter Rosenbrock object...")
LTR1 = CMSDE({**data, **{"compoScheme":"LT", "odeStepScheme":"RB1"}})

print("---------------------------------------------")
print("Creating Strang Rosenbrock object...")
SR1 = CMSDE({**data, **{"compoScheme":"S", "odeStepScheme":"RB1"}})

print("---------------------------------------------")
print("Creating drift-implicit Euler object...")
DIEM = CMSDE({**data,**{"compoScheme":"none","odeStepScheme":"none"}})



if runSimpleDemo:  
    
    path = 0 # Brownian path along which to plot solution 
    comp = 0 # Component of the solution to plot
    
    # Solve system using each method
    t,X_LTdemo = LT.solve()
    _,X_Sdemo = S.solve()
    _,X_LTIEdemo = LTIE.solve()
    _,X_SIEdemo = SIE.solve()   
    _,X_LTR1demo = LTR1.solve()
    _,X_SR1demo = SR1.solve()
    _,X_DIEMdemo = DIEM.solve()
    
    
    # Plot solutions
    fig1 = plt.figure(1)
    ax1 = plt.gca()
    ax1.plot(t[0,:],X_DIEMdemo[path,:,comp],label="$\\tildeX^{DIEM}(t_n)$", color="gray",linestyle="-")
    ax1.plot(t[0,:],X_LTdemo[path,:,comp],label="$\\tildeX^{LT}(t_n)$",color=greenColor,linestyle="-")    
    ax1.plot(t[0,:],X_Sdemo[path,:,comp],label="$\\tildeX^{S}(t_n)$",color=goldColor,linestyle="-")
    ax1.plot(t[0,:],X_LTIEdemo[path,:,comp],label="$\\tildeX^{LTIE}(t_n)$",color=redColor,linestyle="-")    
    ax1.plot(t[0,:],X_SIEdemo[path,:,comp],label="$\\tildeX^{SIE}(t_n)$",color=blueColor,linestyle="-")
    ax1.plot(t[0,:],X_LTR1demo[path,:,comp],label="$\\tildeX^{LTR1}(t_n)$", color=lightRedColor, linestyle="-")
    ax1.plot(t[0,:],X_SR1demo[path,:,comp],label="$\\tildeX^{SR1}(t_n)$", color=lightBlueColor, linestyle="-")    
    ax1.set_xlabel("$t_n$", fontsize=20)
    ax1.set_title(f"[FHN] Oscillatory Dynamics",fontsize=20)
    ax1.set_ylim(-3, 1.5)
    ax1.set_xlim(29,30)
    ax1.tick_params(axis="both", which="major", labelsize=15)
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0+0.1, box1.width, box1.height*0.9])
    leg1 = ax1.legend(loc="lower center",fontsize=15,mode="expand",borderaxespad=0,\
                    fancybox=False, shadow=False, ncol=3)
    leg1.get_frame().set_edgecolor("black")    
    
    


if testMeanSquareConvergence:
    
    # Compute reference solution using Strang splitting 
    _,Xref = S.solve()
    
    # Set step-sizes to test, and measure for testing convergence
    p = np.array([12,11,10,9,8,7,6])
    stepsizes2test = 1/(2**p)
    convergenceMeasure = "meansquare"
    
    # Perform convergence tests
    orderLT, errLT, logStepsizesLT = LT.convergenceTest({"convergenceMeasure":convergenceMeasure, "stepsizes2test":stepsizes2test, "Xref":Xref})
    orderStrang, errStrang, logStepsizesStrang = S.convergenceTest({"convergenceMeasure":convergenceMeasure, "stepsizes2test":stepsizes2test, "Xref":Xref})    
    orderLTIE, errLTIE, logStepsizesLTIE = LTIE.convergenceTest({"convergenceMeasure":convergenceMeasure, "stepsizes2test":stepsizes2test, "Xref":Xref})    
    orderSIE, errSIE, logStepsizesSIE = SIE.convergenceTest({"convergenceMeasure":convergenceMeasure, "stepsizes2test":stepsizes2test, "Xref":Xref})
    orderDIEM, errDIEM, logStepsizesDIEM = DIEM.convergenceTest({"convergenceMeasure":convergenceMeasure, "stepsizes2test":stepsizes2test, "Xref":Xref})
    orderLTR1, errLTR1, logStepsizesLTR1 = LTR1.convergenceTest({"convergenceMeasure":convergenceMeasure, "stepsizes2test":stepsizes2test, "Xref":Xref})
    orderSR1, errSR1, logStepsizesSR1 = SR1.convergenceTest({"convergenceMeasure":convergenceMeasure, "stepsizes2test":stepsizes2test, "Xref":Xref})
    
    print("--------------------------------------------")
    print("LT Errors:       ", errLT)
    print("Strang Errors:   ", errStrang)
    print("LTIE Errors:     ", errLTIE)
    print("SIE Errors:      ", errSIE)   
    print("LTR1 Errors:     ", errLTR1)
    print("SR1 Errors:      ", errSR1)
    print("DIEM Errors:     ", errDIEM)
    
    print("--------------------------------------------")
    print("LT order:       ", orderLT)
    print("Strang order:   ", orderStrang)
    print("LTIE order:     ", orderLTIE)
    print("SIE order:      ", orderSIE)
    print("LTR1 order:     ", orderLTR1)
    print("SR1 order:      ", orderSR1)
    print("DIEM order:     ", orderDIEM)
    
    
    # Plot convergence results for LT, S, LTIE, SIE, DIEM methods
    fig3 = plt.figure(3)
    ax3 = plt.gca()
    ax3.plot(np.exp(logStepsizesStrang), np.exp(logStepsizesStrang), label="$y=h$", color="black", linestyle=":")
    ax3.plot(np.exp(logStepsizesDIEM), np.exp(orderDIEM[0]+orderDIEM[1]*logStepsizesDIEM),color="gray")    
    ax3.scatter(np.exp(logStepsizesDIEM),errDIEM,label="$\\tildeX^{DIEM}(t_n)$", color="gray", marker="p")     
    ax3.plot(np.exp(logStepsizesLT), np.exp(orderLT[0]+orderLT[1]*logStepsizesLT),color=greenColor,linestyle="-")    
    ax3.scatter(np.exp(logStepsizesLT), errLT, label="$\\tildeX^{LT}(t_n)$", color=greenColor, marker = "o")     
    ax3.plot(np.exp(logStepsizesStrang), np.exp(orderStrang[0]+orderStrang[1]*logStepsizesStrang),color=goldColor,linestyle="-")    
    ax3.scatter(np.exp(logStepsizesStrang), errStrang, label="$\\tildeX^{S}(t_n)$", color=goldColor, marker = "s")     
    ax3.plot(np.exp(logStepsizesLTIE), np.exp(orderLTIE[0]+orderLTIE[1]*logStepsizesLTIE),color=redColor,linestyle="--")    
    ax3.scatter(np.exp(logStepsizesLTIE), errLTIE, label="$\\tildeX^{LTIE}(t_n)$", color= redColor, marker = "P")     
    ax3.plot(np.exp(logStepsizesSIE), np.exp(orderSIE[0]+orderSIE[1]*logStepsizesSIE),color=blueColor,linestyle="--")    
    ax3.scatter(np.exp(logStepsizesSIE), errSIE, label="$\\tildeX^{SIE}(t_n)$", color = blueColor, marker = "d") 
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_ylim(10**(-7),0.1)
    ax3.set_xlabel("$h$", fontsize=20)
    ax3.set_ylabel("RMSE($h$)",fontsize=20)
    ax3.tick_params(axis="both", which="major", labelsize=15)
    ax3.set_title(f"[FHN] MSC",fontsize=20)
    box3 = ax3.get_position()
    ax3.set_position([box3.x0+0.03, box3.y0+0.1, box3.width, box3.height*0.9])
    leg3 = ax3.legend(loc="lower center",fontsize=15,mode="expand",borderaxespad=0,\
                    fancybox=False, shadow=False, ncol=3)
    leg3.get_frame().set_edgecolor("black")    
    
    # Plot convergence results for LTIE & SIE vs LT & S
    fig4 = plt.figure(4)
    ax4 = plt.gca()    
    ax4.plot(np.exp(logStepsizesStrang), np.exp(logStepsizesStrang), label="$y=h$", color="black", linestyle="--")    
    ax4.plot(np.exp(logStepsizesLTIE), np.exp(orderLTIE[0]+orderLTIE[1]*logStepsizesLTIE),color=greenColor,linestyle="-")    
    ax4.scatter(np.exp(logStepsizesLTIE), errLTIE, label="$\\tildeX^{LTIE}(t_n)$", color= greenColor, marker = "P")     
    ax4.plot(np.exp(logStepsizesSIE), np.exp(orderSIE[0]+orderSIE[1]*logStepsizesSIE),color=goldColor,linestyle="-")    
    ax4.scatter(np.exp(logStepsizesSIE), errSIE, label="$\\tildeX^{SIE}(t_n)$", color = goldColor, marker = "d") 
    ax4.plot(np.exp(logStepsizesLTR1), np.exp(orderLTR1[0]+orderLTR1[1]*logStepsizesLTR1),color=redColor,linestyle="--")    
    ax4.scatter(np.exp(logStepsizesLTR1), errLTR1, label="$\\tildeX^{LTR1}(t_n)$", color= redColor, marker = "P")     
    ax4.plot(np.exp(logStepsizesSR1), np.exp(orderSR1[0]+orderSR1[1]*logStepsizesSR1),color=blueColor,linestyle="--")    
    ax4.scatter(np.exp(logStepsizesSR1), errSR1, label="$\\tildeX^{SR1}(t_n)$", color = blueColor, marker = "d") 
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_ylim(10**(-7), 0.1)
    ax4.set_xlabel("$h$", fontsize=20)
    ax4.set_ylabel("RMSE($h$)",fontsize=20)
    ax4.tick_params(axis="both", which="major", labelsize=15)
    ax4.set_title(f"[CMP] MSC",fontsize=20)
    box4 = ax4.get_position()
    ax4.set_position([box4.x0+0.03, box4.y0+0.1, box4.width, box4.height*0.9])
    leg4 = ax4.legend(loc="lower center",fontsize=15,mode="expand",borderaxespad=0,\
                    fancybox=False, shadow=False, ncol=3)
    leg4.get_frame().set_edgecolor("black")    
    
    
    
if testErgodicity:
    
    # Compute solutions, mean-square norm of solutions and upper bounds on mean-square norm of solutions
    t, X_LT, XmeanSquareLT, upperBoundLT = LT.ergodicity({"alpha":alpha,"K":K})    
    _, X_S, XmeanSquareS, upperBoundS = S.ergodicity({"alpha":alpha,"K":K})
    _, X_LTIE, XmeanSquareLTIE, upperBoundLTIE = LTIE.ergodicity({"alpha":numAlpha,"K":K})
    _, X_SIE, XmeanSquareSIE, upperBoundSIE = SIE.ergodicity({"alpha":numAlpha,"K":K})
    _, X_DIEM, XmeanSquareDIEM, _ = DIEM.ergodicity({"alpha":alpha,"K":K})
    _, X_LTR1, _, _ = LTR1.ergodicity({"alpha":alpha,"K":K})
    _, X_SR1, _, _ = SR1.ergodicity({"alpha":alpha,"K":K})
    
    # Plot asymptotic bounds --------------------------------------------------------------    
    fig5 = plt.figure(5)
    ax5 = plt.gca()
    ax5.plot(t[0,:],upperBoundLT,color=greenColor,linestyle=":",label="$\\tilde \\kappa^{LT}(t_n)$")
    ax5.plot(t[0,:],upperBoundS,color=goldColor,linestyle=":",label="$\\tilde \\kappa^{S}(t_n)$")
    ax5.plot(t[0,:],upperBoundLTIE,color=redColor,linestyle=":",label="$\\tilde \\kappa^{LTIE}(t_n)$")
    ax5.plot(t[0,:],upperBoundSIE,color=blueColor,linestyle=":",label="$\\tilde \\kappa^{SIE}(t_n)$")
    ax5.plot(t[0,:],XmeanSquareLT,color=greenColor,linestyle="-",label="E$[||{\\tildeX^{LT}}(t_n)||^2]$")    
    ax5.plot(t[0,:],XmeanSquareS,color=goldColor,linestyle="-",label="E$[||{\\tildeX^{S}}(t_n)||^2]$")    
    ax5.plot(t[0,:],XmeanSquareLTIE,color=redColor,linestyle="-",label="E$[||{\\tildeX^{LTIE}}(t_n)||^2]$")   
    ax5.plot(t[0,:],XmeanSquareSIE,color=blueColor,linestyle="-",label="E$[||{\\tildeX^{SIE}}(t_n)||^2]$")  
    ax5.set_xlabel("$t_n$", fontsize=20)
    ax5.set_title(f"[FHN] Asymptotic Bounds",fontsize=20)
    ax5.set_ylim(-40, 40)
    ax5.tick_params(axis="both", which="major", labelsize=15)
    box5 = ax5.get_position()
    ax5.set_position([box5.x0, box5.y0+0.1, box5.width, box5.height*0.9])
    leg5 = ax5.legend(loc="lower center",fontsize=15,mode="expand",borderaxespad=0,\
                    fancybox=False, shadow=False, ncol=2)
    leg5.get_frame().set_edgecolor("black")  
    
    
    # Plot histograms of solution for each method -----------------------------------------
    
    # Component to plot
    comp = 0
    
    # Lie-Trotter splitting method histogram
    fig6 = plt.figure(6)
    ax6 = plt.gca()
    ax6.hist(X_LT[:,-1,comp],bins=50,density=True,color=greenColor,label="$\\tildeX^{LT}_1(t_n)$")
    ax6.set_xlabel("$x$",fontsize=20)
    ax6.set_title("[FHN] Density of $\\tildeX^{LT}_1(t_n)$",fontsize=25)
    ax6.set_xlim(-2,2)
    ax6.set_ylim(0,2)
    ax6.tick_params(axis="both",which="major",labelsize=15)
    box6 = ax6.get_position()
    ax6.set_position([box6.x0,box6.y0+0.05,box6.width,box6.height*0.9])
    leg6 = ax6.legend(loc="upper right", fontsize=15)
    leg6.get_frame().set_edgecolor("black")
    
    # Strang splitting method histogram
    fig7 = plt.figure(7)
    ax7 = plt.gca()
    ax7.hist(X_S[:,-1,comp],bins=50,density=True,color=goldColor,label="$\\tildeX^{S}_1(t_n)$")
    ax7.set_xlabel("$x$",fontsize=20)
    ax7.set_title("[FHN] Density of $\\tildeX^{S}_1(t_n)$",fontsize=25)
    ax7.set_xlim(-2,2)
    ax7.set_ylim(0,2)
    ax7.tick_params(axis="both",which="major",labelsize=15)
    box7 = ax7.get_position()
    ax7.set_position([box7.x0,box7.y0+0.05,box7.width,box7.height*0.9])
    leg7 = ax7.legend(loc="upper right", fontsize=15)
    leg7.get_frame().set_edgecolor("black")
    
    # Lie-Trotter Implicit Euler (LTIE) composition method histogram
    fig8 = plt.figure(8)
    ax8 = plt.gca()
    ax8.hist(X_LTIE[:,-1,comp],bins=50,density=True,color=redColor,label="$\\tildeX^{LTIE}_1(t_n)$")
    ax8.set_xlabel("$x$",fontsize=20)
    ax8.set_title("[FHN] Density of $\\tildeX^{LTIE}_1(t_n)$",fontsize=25)
    ax8.set_xlim(-2,2)
    ax8.set_ylim(0,2)
    ax8.tick_params(axis="both",which="major",labelsize=15)
    box8 = ax8.get_position()
    ax8.set_position([box8.x0,box8.y0+0.05,box8.width,box8.height*0.9])
    leg8 = ax8.legend(loc="upper right", fontsize=15)
    leg8.get_frame().set_edgecolor("black")
    
    # Strang implicit Euler (SIE) composition method histogram
    fig9 = plt.figure(9)
    ax9 = plt.gca()
    ax9.hist(X_SIE[:,-1,comp],bins=50,density=True,color=blueColor,label="$\\tildeX^{SIE}_1(t_n)$")
    ax9.set_xlabel("$x$",fontsize=20)
    ax9.set_title("[FHN] Density of $\\tildeX^{SIE}_1(t_n)$",fontsize=25)
    ax9.set_xlim(-2,2)
    ax9.set_ylim(0,2)
    ax9.tick_params(axis="both",which="major",labelsize=15)
    box9 = ax9.get_position()
    ax9.set_position([box9.x0,box9.y0+0.05,box9.width,box9.height*0.9])
    leg9 = ax9.legend(loc="upper right", fontsize=15)
    leg9.get_frame().set_edgecolor("black")
    
    # Drift-implicit Euler-Maruyama method histogram
    fig10 = plt.figure(10)
    ax10 = plt.gca()
    ax10.hist(X_DIEM[:,-1,comp],bins=50,density=True,color="grey",label="$\\tildeX^{DIEM}_1(t_n)$")
    ax10.set_xlabel("$x$",fontsize=20)
    ax10.set_title("[FHN] Density of $\\tildeX^{DIEM}_1(t_n)$",fontsize=25)
    ax10.set_xlim(-2,2)
    ax10.set_ylim(0,2)
    ax10.tick_params(axis="both",which="major",labelsize=15)
    box10 = ax10.get_position()
    ax10.set_position([box10.x0,box10.y0+0.05,box10.width,box10.height*0.9])
    leg10 = ax10.legend(loc="upper right", fontsize=15)
    leg10.get_frame().set_edgecolor("black")
    
    # Lie-Trotter first-order Rosenbrock composition method histogram
    fig11 = plt.figure(11)
    ax11 = plt.gca()
    ax11.hist(X_LTR1[:,-1,comp],bins=50,density=True,color=lightRedColor,label="$\\tildeX^{LTR1}_1(t_n)$")
    ax11.set_xlabel("$x$",fontsize=20)
    ax11.set_title("[FHN] Density of $\\tildeX^{LTR1}_1(t_n)$",fontsize=25)
    ax11.set_xlim(-2,2)
    ax11.set_ylim(0,2)    
    ax11.tick_params(axis="both",which="major",labelsize=15)
    box11 = ax11.get_position()
    ax11.set_position([box11.x0,box11.y0+0.05,box11.width,box11.height*0.9])
    leg11 = ax11.legend(loc="upper right", fontsize=15)
    leg11.get_frame().set_edgecolor("black")
    
    # Strang first-order Rosenbrock composition method histogram
    fig12 = plt.figure(12)
    ax12 = plt.gca()
    ax12.hist(X_SR1[:,-1,comp],bins=50,density=True,color=lightBlueColor,label="$\\tildeX^{SR1}_1(t_n)$")
    ax12.set_xlabel("$x$",fontsize=20)
    ax12.set_title("[FHN] Density of $\\tildeX^{SR1}_1(t_n)$",fontsize=25)
    ax12.set_xlim(-2,2)
    ax12.set_ylim(0,2)
    ax12.tick_params(axis="both",which="major",labelsize=15)
    box12 = ax12.get_position()
    ax12.set_position([box12.x0,box12.y0+0.05,box12.width,box12.height*0.9])
    leg12 = ax12.legend(loc="upper right", fontsize=15)
    leg12.get_frame().set_edgecolor("black")
    
    
    
plt.show()


    
    


