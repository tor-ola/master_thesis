import sys
sys.path.append("/Users/TorOla/Desktop/NTNU/Master Thesis/Python/Classes")
from CMSDE import CMSDE
from CMSDE import generateRandomVariables
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

runSimpleDemo = False
testMeanSquareConvergence = False
testErgodicity = True

# Exact solution of nonlinear ODE
def exactStep(x,h,eqParams):
    return x/np.sqrt(1+2*x**2*h)

# Right-hand-side of nonlinear ODE
def B(x,eqParams):
    return -x**3
    
# Jacobian of B
def JB(x,eqParams):
    M = eqParams["M"]
    d = eqParams["d"]
    J = np.zeros((M,d,d),dtype=float)
    J[:,0,0] = -3*x[:,0]**2
    return J

# Exact solution of implicit equation
def implicitExactStep(x,h,eqParams):
    a = 1/h
    b = x/h
    y = ((-b+np.sqrt(b**2+4*(a/3)**3))/2)**(1/3)
    z = a/(3*y)
    return z-y

# Rosenbrock approximation to solution of nonlinear ODE
def rosenbrockStep(x,h,eqParams):
    return x-(h*x**3)/(1+3*h*x**2)
   
# Full drift term 
def F(x,eqParams):
    omega = eqParams["omega"]
    return omega*x-x**3
    
# Jacobian of F
def JF(x,eqParams):
    omega = eqParams["omega"]
    M = eqParams["M"]
    d = eqParams["d"]
    J = np.zeros((M,d,d),dtype=float)
    J[:,0,0] = omega-3*x[:,0]**2
    return J

# Exact solution to implicit equation associated with the drift-implicit Euler-Maruyama method. 
def DIEMimplicitExactStep(x,h,eqParams,dW):
    omega = eqParams["omega"]
    a = (1-h*omega)/h
    b = (x+dW)/h
    y = ((-b+np.sqrt(b**2+4*(a/3)**3))/2)**(1/3)
    z = a/(3*y)
    return z-y

    
# Set Parameters 
sig1 = 1.0 # Noise
sig = np.array([[sig1]]) # Diffusion matrix
omega = 1.0 # omega parameter
t0 = 0.0 # start time
T = 20.0 # end time 
h = 0.01 # step-size
N = int((T-t0)/h) # Number of discrete time-points
print("Number of discrete time-points:", N)
M = 1000 # Number of simulated Brownian paths
X0 = np.array([2.0]) # Initial value
A = np.array([[omega]]) # Linear part of the drift
d = np.shape(A)[0] # dimensionality of system
mu = np.max(np.linalg.eigvals(0.5*(A+A.T))) # Logarithmic norm of A
print(f"Logarithmic norm of A: {mu}")

# Value of alpha from Assumption 3.3 (see report)
alpha = 2.0
numAlpha = 1/(2*h)*np.log(1+2*alpha*h) # numerical alpha (in report: \tilde{\alpha}_0 )
print(f"Alpha:              {alpha}")
print(f"Numerical alpha:    {numAlpha}")

# Value of K from Assumption 3.3 (see report)
if alpha > 0:
    K = alpha**2/4
else:
    K = 0.0
    
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
eqParams = {"A": A, "sig": sig, "omega":omega, "M":M,"d":d}

# Input data for our nuumerical methods
data = {"eqParams":eqParams,"X0":X0,"M":M, "h":h, "t0":t0, "T":T,"B":B, "JB":JB,\
        "F":F, "JF":JF,"Y":Y, "dW":dW,\
        "exactStep":exactStep,"implicitExactStepIsDefined":True, "implicitExactStep":implicitExactStep,\
        "DIEMimplicitExactStepIsDefined":True, "DIEMimplicitExactStep":DIEMimplicitExactStep,\
        "rosenbrockStep":rosenbrockStep,"randomInit":True}

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
    
    comp = 0 # Brownian path along which to plot solution 
    path = 0 # Component of the solution to plot
    
    # Solve system using each method
    t,X_LTdemo = LT.solve()
    _,X_Sdemo = S.solve()
    _,X_LTIEdemo = LTIE.solve()
    _,X_SIEdemo = SIE.solve()    
    _,X_LTR1demo = LTR1.solve()
    _,X_SR1demo = SR1.solve()
    _,X_DIEMdemo = DIEM.solve()
                
    # Plot solutions
    fig1 = plt.figure(1,constrained_layout=True)
    ax1 = plt.subplot(111)
    ax1.plot(t[0,:],X_DIEMdemo[path,:,comp],label="$\\tildeX^{DIEM}(t_n)$", color="gray",linestyle="-")
    ax1.plot(t[0,:],X_LTdemo[path,:,comp],label="$\\tildeX^{LT}(t_n)$",color=greenColor,linestyle="-")    
    ax1.plot(t[0,:],X_Sdemo[path,:,comp],label="$\\tildeX^{S}(t_n)$",color=goldColor,linestyle="-")
    ax1.plot(t[0,:],X_LTIEdemo[path,:,comp],label="$\\tildeX^{LTIE}(t_n)$",color=redColor,linestyle="--")    
    ax1.plot(t[0,:],X_SIEdemo[path,:,comp],label="$\\tildeX^{SIE}(t_n)$",color=blueColor,linestyle="--") 
    ax1.plot(t[0,:],X_LTR1demo[path,:,comp], label="$\\tildeX^{LTR1}(t_n)$", color=lightRedColor,linestyle=":")
    ax1.plot(t[0,:],X_SR1demo[path,:,comp],label="$\\tildeX^{SR1}(t_n)$", color=lightBlueColor, linestyle=":")
    ax1.set_xlim(-0.2,5)
    ax1.set_ylim(-3.5,5.0)
    ax1.set_xlabel("$t$",fontsize=20)
    ax1.set_ylabel("$\\tildeX(t_n)$",fontsize=20)
    ax1.tick_params(axis="both", which="major", labelsize=15)
    ax1.set_title(f"$X_0$ = {X0[0]:.0e}, $\\omega$={omega}",fontsize=25)
    box1 = ax1.get_position()
    ax1.set_position([box1.x0+0.02, box1.y0+0.1, box1.width, box1.height*0.9])
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
    ax3.set_ylim(10**(-9),0.1)
    ax3.set_xlabel("$h$", fontsize=20)
    ax3.set_ylabel("RMSE($h$)",fontsize=20)
    ax3.tick_params(axis="both", which="major", labelsize=15)
    ax3.set_title(f"[CMP] MSC ($\\omega$={omega:.1f}, $\\sigma$={sig1:.1f})",fontsize=20)
    box3 = ax3.get_position()
    ax3.set_position([box3.x0+0.03, box3.y0+0.1, box3.width, box3.height*0.9])
    leg3 = ax3.legend(loc="lower center",fontsize=15,mode="expand",borderaxespad=0,\
                    fancybox=False, shadow=False, ncol=3)
    leg3.get_frame().set_edgecolor("black")    
    
    # Plot convergence results for LTIE & SIE vs LT & S
    fig4 = plt.figure(4)
    ax4 = plt.gca()    
    ax4.plot(np.exp(logStepsizesStrang), np.exp(logStepsizesStrang), label="$y=h$", color="black", linestyle=":")    
    ax4.plot(np.exp(logStepsizesLTIE), np.exp(orderLTIE[0]+orderLTIE[1]*logStepsizesLTIE),color=greenColor,linestyle="-")    
    ax4.scatter(np.exp(logStepsizesLTIE), errLTIE, label="$\\tildeX^{LTIE}(t_n)$", color= greenColor, marker = "o")     
    ax4.plot(np.exp(logStepsizesSIE), np.exp(orderSIE[0]+orderSIE[1]*logStepsizesSIE),color=goldColor,linestyle="-")    
    ax4.scatter(np.exp(logStepsizesSIE), errSIE, label="$\\tildeX^{SIE}(t_n)$", color = goldColor, marker = "s") 
    ax4.plot(np.exp(logStepsizesLTR1), np.exp(orderLTR1[0]+orderLTR1[1]*logStepsizesLTR1),color=redColor,linestyle="--")    
    ax4.scatter(np.exp(logStepsizesLTR1), errLTR1, label="$\\tildeX^{LTR1}(t_n)$", color= redColor, marker = "p")     
    ax4.plot(np.exp(logStepsizesSR1), np.exp(orderSR1[0]+orderSR1[1]*logStepsizesSR1),color=blueColor,linestyle="--")    
    ax4.scatter(np.exp(logStepsizesSR1), errSR1, label="$\\tildeX^{SR1}(t_n)$", color = blueColor, marker = "d") 
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_ylim(10**(-9), 0.1)
    ax4.set_xlabel("$h$", fontsize=20)
    ax4.set_ylabel("RMSE($h$)",fontsize=20)
    ax4.tick_params(axis="both", which="major", labelsize=15)
    ax4.set_title(f"[CMP] MSC ($\\omega$={omega:.1f}, $\\sigma$={sig1:.1f})",fontsize=20)
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
    ax5.set_title(f"[CMP] Asymptotic Bounds ($\\omega$={omega})",fontsize=20)
    ax5.set_ylim(-4.5, 4.5)
    ax5.tick_params(axis="both", which="major", labelsize=15)
    box5 = ax5.get_position()
    ax5.set_position([box5.x0, box5.y0+0.1, box5.width, box5.height*0.9])
    leg5 = ax5.legend(loc="lower center",fontsize=15,mode="expand",borderaxespad=0,\
                    fancybox=False, shadow=False, ncol=2)
    leg5.get_frame().set_edgecolor("black")    
    
   
    # Compute Gibbs distribution 
    sampleIndex = -1
    if sampleIndex < 0:
        sampleTime = T+(sampleIndex+1)*h
    else:
        sampleTime = sampleIndex*h
    samplePoints = np.linspace(-5,5,5000)    
    gibbsDistKernel = np.exp((omega*samplePoints**2-0.5*samplePoints**4)/sig1**2)
    gibbsDistLambda = lambda x: np.exp((omega*x**2-0.5*x**4)/sig1**2)
    I = integrate.quadrature(gibbsDistLambda,-5,5,maxiter=200)[0]
    gibbsDist = gibbsDistKernel/I
    print(I)
        
    # Plot histograms of solution for each method -----------------------------------------
    
    # Lie-Trotter splitting method histogram
    fig6 = plt.figure(6)
    ax6 = plt.gca()
    ax6.plot(samplePoints,gibbsDist,color="black", linestyle="-", label="Gibbs")
    ax6.hist(X_LT[:,sampleIndex,0],bins=50,density=True,color=greenColor,label="$\\tildeX^{LT}(t_n)$")
    ax6.set_xlabel("$x$",fontsize=20)
    ax6.set_title("[CMP] Density of $\\tildeX^{LT}(t_n)$",fontsize=25)
    ax6.set_ylim(0,0.8)
    ax6.tick_params(axis="both",which="major",labelsize=15)
    box6 = ax6.get_position()
    ax6.set_position([box6.x0,box6.y0+0.05,box6.width,box6.height*0.9])
    leg6 = ax6.legend(loc="upper left", fontsize=15)
    leg6.get_frame().set_edgecolor("black")
    
    # Strang splitting method histogram
    fig7 = plt.figure(7)
    ax7 = plt.gca()
    ax7.plot(samplePoints,gibbsDist,color="black", linestyle="-", label="Gibbs")
    ax7.hist(X_S[:,sampleIndex,0],bins=50,density=True,color=goldColor,label="$\\tildeX^{S}(t_n)$")
    ax7.set_xlabel("$x$",fontsize=20)
    ax7.set_title("[CMP] Density of $\\tildeX^{S}(t_n)$",fontsize=25)
    ax7.set_ylim(0,0.8)
    ax7.tick_params(axis="both",which="major",labelsize=15)
    box7 = ax7.get_position()
    ax7.set_position([box7.x0,box7.y0+0.05,box7.width,box7.height*0.9])
    leg7 = ax7.legend(loc="upper left", fontsize=15)
    leg7.get_frame().set_edgecolor("black")
    
    # Lie-Trotter Implicit Euler (LTIE) composition method histogram
    fig8 = plt.figure(8)
    ax8 = plt.gca()
    ax8.plot(samplePoints,gibbsDist,color="black", linestyle="-", label="Gibbs")
    ax8.hist(X_LTIE[:,sampleIndex,0],bins=50,density=True,color=redColor,label="$\\tildeX^{LTIE}(t_n)$")
    ax8.set_xlabel("$x$",fontsize=20)
    ax8.set_title("[CMP] Density of $\\tildeX^{LTIE}(t_n)$",fontsize=25)
    ax8.set_ylim(0,0.8)
    ax8.tick_params(axis="both",which="major",labelsize=15)
    box8 = ax8.get_position()
    ax8.set_position([box8.x0,box8.y0+0.05,box8.width,box8.height*0.9])
    leg8 = ax8.legend(loc="upper left", fontsize=15)
    leg8.get_frame().set_edgecolor("black")
    
    # Strang implicit Euler (SIE) composition method histogram
    fig9 = plt.figure(9)
    ax9 = plt.gca()
    ax9.plot(samplePoints,gibbsDist,color="black", linestyle="-", label="Gibbs")
    ax9.hist(X_SIE[:,sampleIndex,0],bins=50,density=True,color=blueColor,label="$\\tildeX^{SIE}(t_n)$")
    ax9.set_xlabel("$x$",fontsize=20)
    ax9.set_title("[CMP] Density of $\\tildeX^{SIE}(t_n)$",fontsize=25)
    ax9.set_ylim(0,0.8)
    ax9.tick_params(axis="both",which="major",labelsize=15)
    box9 = ax9.get_position()
    ax9.set_position([box9.x0,box9.y0+0.05,box9.width,box9.height*0.9])
    leg9 = ax9.legend(loc="upper left", fontsize=15)
    leg9.get_frame().set_edgecolor("black")
    
    # Drift-implicit Euler-Maruyama method histogram
    fig10 = plt.figure(10)
    ax10 = plt.gca()
    ax10.plot(samplePoints,gibbsDist,color="black", linestyle="-", label="Gibbs")
    ax10.hist(X_DIEM[:,sampleIndex,0],bins=50,density=True,color="grey",label="$\\tildeX^{DIEM}(t_n)$")
    ax10.set_xlabel("$x$",fontsize=20)
    ax10.set_title("[CMP] Density of $\\tildeX^{DIEM}(t_n)$",fontsize=25)
    ax10.set_ylim(0,0.8)
    ax10.tick_params(axis="both",which="major",labelsize=15)
    box10 = ax10.get_position()
    ax10.set_position([box10.x0,box10.y0+0.05,box10.width,box10.height*0.9])
    leg10 = ax10.legend(loc="upper left", fontsize=15)
    leg10.get_frame().set_edgecolor("black")
    
    # Lie-Trotter first-order Rosenbrock composition method histogram
    fig11 = plt.figure(11)
    ax11 = plt.gca()
    ax11.plot(samplePoints,gibbsDist,color="black", linestyle="-", label="Gibbs")
    ax11.hist(X_LTR1[:,sampleIndex,0],bins=50,density=True,color=lightRedColor,label="$\\tildeX^{LTR1}(t_n)$")
    ax11.set_xlabel("$x$",fontsize=20)
    ax11.set_title("[CMP] Density of $\\tildeX^{LTR1}(t_n)$",fontsize=25)
    ax11.set_ylim(0,0.8)
    ax11.tick_params(axis="both",which="major",labelsize=15)
    box11 = ax11.get_position()
    ax11.set_position([box11.x0,box11.y0+0.05,box11.width,box11.height*0.9])
    leg11 = ax11.legend(loc="upper left", fontsize=15)
    leg11.get_frame().set_edgecolor("black")
    
    # Strang first-order Rosenbrock composition method histogram
    fig12 = plt.figure(12)
    ax12 = plt.gca()
    ax12.plot(samplePoints,gibbsDist,color="black", linestyle="-", label="Gibbs")
    ax12.hist(X_SR1[:,sampleIndex,0],bins=50,density=True,color=lightBlueColor,label="$\\tildeX^{SR1}(t_n)$")
    ax12.set_xlabel("$x$",fontsize=20)
    ax12.set_title("[CMP] Density of $\\tildeX^{SR1}(t_n)$",fontsize=25)
    ax12.set_ylim(0,0.8)
    ax12.tick_params(axis="both",which="major",labelsize=15)
    box12 = ax12.get_position()
    ax12.set_position([box12.x0,box12.y0+0.05,box12.width,box12.height*0.9])
    leg12 = ax12.legend(loc="upper left", fontsize=15)
    leg12.get_frame().set_edgecolor("black")
    


plt.show()


