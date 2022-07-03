import numpy as np 
from numpy import matlib as mb
import scipy.linalg as la 


class CMSDE:

# --- Constructor --- #
    def __init__(self,data):
        
        # Parse input data
        if isinstance(data,dict):
            self.compoScheme = ife("compoScheme" in data, data["compoScheme"], "Error: Please specify a composition scheme.")
            self.odeStepScheme = ife("odeStepScheme" in data, data["odeStepScheme"], "Error: Please specify an ODE solver step.")
            self.h = data["h"] if "h" in data else 2**(-10)
            self.t0 = data["t0"] if "t0" in data else 0.0
            self.T = data["T"] if "T" in data else 1.0
            self.N = int((self.T-self.t0)/self.h)              
            self.M = data["M"] if "M" in data else 1000
            self.eqParams = ife("eqParams" in data, data["eqParams"], "Error: Equation parameters 'eqParams' not supplied.")
            self.A = ife("A" in self.eqParams, self.eqParams["A"], "Error: Matrix A not supplied to 'eqParams'.")
            self.sig = ife("sig" in self.eqParams, self.eqParams["sig"], "Error: Matrix Sigma not supplied to 'eqParams'.")
            self.d = np.shape(self.A)[0]
            self.X0 = data["X0"] if "X0" in data else np.zeros(self.d,dtype=float)
            t = np.linspace(self.t0,self.T,self.N+1)
            self.t = mb.repmat(t,self.M,1)
            self.B = ife("B" in data, data["B"], "Error: Nonlinear term 'B' not supplied.")
            self.JB = ife("JB" in data, data["JB"], "Error: Jacobian of 'B' not supplied.")
            self.exactStep = ife("exactStep" in data, data["exactStep"], "Error: Exact solution step of nonlinear term not supplied.")
            self.F = ife("F" in data, data["F"], "Error: Full drift 'F' not supplied.")
            self.JF = ife("JF" in data, data["JF"], "Error: Jacobian of full drift 'F' not supplied.")
            self.Y = ife("Y" in data, data["Y"], "Error: Please supply random variables 'Y'.")
            self.dW = ife("dW" in data, data["dW"], "Error: Please supply random variables 'dW'.")
            
            self.implicitExactStepIsDefined = data["implicitExactStepIsDefined"] if "implicitExactStepIsDefined" in data else False
            if self.implicitExactStepIsDefined:
                self.implicitExactStep = data["implicitExactStep"]
            else:
                self.implicitExactStep = None
            
            self.DIEMimplicitExactStepIsDefined = data["DIEMimplicitExactStepIsDefined"] if "DIEMimplicitExactStepIsDefined" in data else False
            if self.DIEMimplicitExactStepIsDefined:
                self.DIEMimplicitExactStep = data["DIEMimplicitExactStep"]
            else:
                self.DIEMimplicitExactStep = None
            
            self.rosenbrockStep = data["rosenbrockStep"] if "rosenbrockStep" in data else None
            self.mu = np.max(np.linalg.eigvals(0.5*(self.A+(self.A).T)))
            self.matExp = la.expm(self.A*self.h)
            self.randomInit = data["randomInit"] if "randomInit" in data else False
            
            
        
     # --- General solver --- #
    def solve(self):       
        
        # Set initial value of solution
        X = self.initX({"N":self.N})        
        
        # Run solver
        if self.compoScheme == "LT":
            print("---------------------------------------------------")
            print(f"Solving system using Lie-Trotter composition with {self.odeStepScheme} solution of nonlinear ODE.")
            for n in range(1,self.N+1):
                X[:,n,:] = self.lieTrotterStep(X[:,n-1,:],self.Y[:,n-1,:],self.h,self.matExp)
        elif self.compoScheme == "S":
            print("---------------------------------------------------")
            print(f"Solving system using Strang composition with {self.odeStepScheme} solution of nonlinear ODE.")
            for n in range(1,self.N+1):
                X[:,n,:] = self.strangStep(X[:,n-1,:],self.Y[:,n-1,:],self.h,self.matExp)
        elif self.compoScheme == "none":
            print("---------------------------------------------------")
            print(f"Solving system using drift-implicit Euler Maruyama method (No equation splitting).")
            for n in range(1,self.N+1):
                X[:,n,:] = self.DIEMstep(X[:,n-1,:],self.dW[:,n-1,:],self.h)
        else:
            raise Exception("Invalid choice of splitting / composition scheme. Choose 'LT' (Lie-Trotter), 'S' (Strang) or 'none' (drift implicit Euler with no equation splitting).")
        return self.t,X
    
     # --- Lie Trotter solver step --- #
    def lieTrotterStep(self,x,y,h,matExp):
        if self.odeStepScheme == "exact":
            x = self.exactStep(x,h,self.eqParams)
        elif self.odeStepScheme == "IE":
            x = self.implicitEulerStep(x,h)
        elif self.odeStepScheme == "RB1":
            x = self.rosenbrockStep(x,h,self.eqParams)
        else:
            raise Exception("Invalid choice of ODE solver. Set 'odeStepScheme' equal to 'exact', 'IE' (Implicit Euler) or 'RB1' (First-order Rosenbrock)")
        x = np.einsum('ij,mj->mi',matExp,x,optimize=True)+y
        return x
        
    # --- Strang solver step --- #
    def strangStep(self,x,y,h,matExp):
        if self.odeStepScheme == "exact":
            x = self.exactStep(x,h/2,self.eqParams)
            x = np.einsum('ij,mj->mi',matExp,x,optimize=True) + y
            x = self.exactStep(x,h/2,self.eqParams)
        elif self.odeStepScheme == "IE":
            x = self.implicitEulerStep(x,h/2)
            x = np.einsum('ij,mj->mi',matExp,x,optimize=True) + y
            x = self.implicitEulerStep(x,h/2)
        elif self.odeStepScheme == "RB1":
            x = self.rosenbrockStep(x,h/2,self.eqParams)
            x = np.einsum('ij,mj->mi',matExp,x,optimize=True) + y
            x = self.rosenbrockStep(x,h/2,self.eqParams)
        else:
            raise Exception("Invalid choice of ODE solver. Set 'odeStepScheme' equal to 'exact', 'IE' (Implicit Euler) or 'RB1' (First-order Rosenbrock)")
        return x
        
    # --- Implicit Euler solver step --- #
    def implicitEulerStep(self,x,h):
        if self.implicitExactStepIsDefined:
            return self.implicitExactStep(x,h,self.eqParams)
        else:
            TOL = 1e-9
            itermax = 100
            k = 0
            xnm1 = x
            d = self.d
            while k < itermax:
                J = self.JB(x,self.eqParams)
                newtonMatrix = np.identity(d,dtype=float) - h*J
                rhs =  xnm1 + h*self.B(x,self.eqParams) - h*np.einsum('mij,mj->mi',J,x)
                xnew = np.linalg.solve(newtonMatrix,rhs)
                errs = la.norm(xnew-x,ord=2,axis=-1)
                if np.max(errs)<TOL:
                    return xnew
                else:
                    k += 1
                    x = xnew
        print(f"Newton's method failed to converge to tolerance {TOL} within {itermax} iterations. Results may be inaccurate. Max error = {np.max(errs)}.")
        return xnew
                
    
    # Drift-implicit Euler Maruyama step
    def DIEMstep(self,x,dW,h):
        if self.DIEMimplicitExactStepIsDefined:
            return self.DIEMimplicitExactStep(x,h,self.eqParams,dW)
        else:
            TOL= 1e-9
            itermax = 100
            k = 0
            xnm1 = x
            d = self.d
            while k < itermax:
                J = self.JF(x,self.eqParams)
                newtonMatrix = np.identity(d,dtype=float)-h*J
                rhs = xnm1 + h*self.F(x,self.eqParams) + dW - h*np.einsum('mij,mj->mi',J,x)
                xnew = np.linalg.solve(newtonMatrix,rhs)
                errs = la.norm(xnew-x,ord=2,axis=-1)
                if np.max(errs) < TOL:
                    return xnew
                else:
                    k += 1
                    x = xnew
        print(f"Newton's method failed to converge to tolerance {TOL} within {itermax} iterations. Results may be inaccurate. Max error = {np.max(errs)}.")
        return xnew
            
    
    # --- Initialize solution --- #
    def initX(self,data):
        N = data["N"]
        X = np.zeros((self.M,N+1,self.d),dtype=float)
        if self.randomInit:
            np.random.seed(69)
            X[0:self.M,0,0:self.d] = np.random.uniform(1.0,2.0,(self.M,self.d))
        else:
            X[0:self.M,0,0:self.d] = self.X0
        return X
    
    # --- Compute Convergence Order --- #
    def getConvergenceOrder(self,err,logH,numTests):
        D = np.ones((numTests,2),dtype=float)
        D[:,1] = logH
        rhs = np.log(err)
        q = la.lstsq(D,rhs)
        return q[0]
    
    
    # Test convergence
    def convergenceTest(self,data):
        
        print("------------------------------------------------------------")
        
        stepsizes2test = ife("stepsizes2test" in data, data["stepsizes2test"], "Error: Please specify stepsizes to test")
        convergenceMeasure = ife("convergenceMeasure" in data, data["convergenceMeasure"],"Error: Please specify how convergence is to be measured ('meansquare' or 'strong').")
        Xtrue = ife("Xref" in data, data["Xref"], "Error: Please specify a reference solution")
        H = data["H"] if "H" in data else 0.0
        
        print("Testing",convergenceMeasure,"convergence of",self.compoScheme,"using",self.odeStepScheme, "solution of nonlinear ODE")
        print("stepsizes:",stepsizes2test)
        
        err = np.zeros(len(stepsizes2test),dtype=float)
        logStepsizes = np.zeros(len(stepsizes2test),dtype=float)
        k = 0
        
        # Compute error of solution for each step-size to be tested
        for stepsize in stepsizes2test:
            
            # Ensure that fine grid parameters are integer multiples of coarse grid parameters 
            R = stepsize/self.h
            L = self.N/R
            if R % int(R) == 0.0 and L % int(L) == 0.0:
                R = int(R)
                L = int(L)
            else:
                print("R =",R)
                print("L =",L)
                raise Exception("One or more fine grid parameters (number of sample points, stepsizes, etc.) are not integer multiples of coarse grid parameters.")
            
            
            print("Computing solution for stepsize",stepsize,"(L =",L,")")
            logStepsizes[k] = np.log(stepsize)
            matExp = la.expm(self.A*stepsize) # Matrix exponential of Ah
            X = self.initX({"N":L})
            
            # Used to produce correct Brownian paths at larger step-sizes for splitting/composition methods
            correctionTerm = np.zeros((R,self.d,self.d),dtype=float)
            for r in range(R):
                correctionTerm[r,:] = la.expm(self.A*(R-r-1)*self.h)
                
            # Test splitting / composition methods
            if self.compoScheme == "S":
                for l in range(1,L+1):
                    y = np.einsum('nij,mnj->mi',correctionTerm,self.Y[:,(l-1)*R:l*R,:],optimize=True) 
                    X[:,l,:] = self.strangStep(X[:,l-1,:],y,stepsize,matExp)
            elif self.compoScheme == "LT":
                for l in range(1,L+1):
                    y = np.einsum('nij,mnj->mi',correctionTerm,self.Y[:,(l-1)*R:l*R,:],optimize=True) 
                    X[:,l,:] = self.lieTrotterStep(X[:,l-1,:],y,stepsize,matExp)
            # Test drift-implicit Euler Maruyama method
            elif self.compoScheme == "none":
                for l in range(1,L+1):
                    dW = np.sum(self.dW[:,(l-1)*R:l*R,:],1)
                    X[:,l,:] = self.DIEMstep(X[:,l-1,:],dW,stepsize)
                        
            if abs(H - stepsize) < 1e-12:
                Xcoarse = X
                    
            # Compute errors according to the selected convergence measure
            Xnorm = la.norm(X[:,-1:,:]-Xtrue[:,-1:,:],ord=2,axis=-1)
            if convergenceMeasure == "meansquare":
                err[k] = np.sqrt(np.mean(Xnorm**2)) 
            elif convergenceMeasure == "strong":
                err[k] = np.mean(Xnorm) 
            else:
                raise Exception("Invalid choice of convergence measure; please choose 'meansquare' or 'strong'.")
               
                
            # Increment counter 
            k += 1
        res = self.getConvergenceOrder(err,logStepsizes,len(stepsizes2test))
        
        if H == 0.0:
            return res,err,logStepsizes
        else:
            return res,err,logStepsizes,Xcoarse
    
    
    def ergodicity(self,data):
        
        # Compute solution and mean-square norm of solution
        t,X = self.solve()
        XmeanSquare= np.mean(np.linalg.norm(X,ord=2,axis=-1)**2,axis=0)
        
        # Constants 'alpha' and 'K' from Assumption 3.3 (see report)
        alpha = ife("alpha" in data, data["alpha"], "Error: A value for alpha must be supplied.")
        K = ife("K" in data, data["K"], "Error: A value for K must be supplied.")
        
        # Squared Frobenius norm of diffusion matrix 
        sigNorm = np.linalg.norm(self.sig,ord="fro")**2
        
        # Compute upper bounds on mean-square norm of solution for the given method
        if self.compoScheme == "LT" and self.odeStepScheme == "exact":
            C = np.exp(2*self.mu*self.h)*(2*K*thetaFunction(alpha*self.h) + sigNorm*thetaFunction(self.mu*self.h))
        elif self.compoScheme == "S" and self.odeStepScheme == "exact":
            C = K*(np.exp((2*self.mu-alpha)*self.h)+1)*thetaFunction(0.5*alpha*self.h) + sigNorm*np.exp((2*self.mu-alpha)*self.h)*thetaFunction(self.mu*self.h)
        elif self.compoScheme == "LT" and self.odeStepScheme == "IE":
            C = np.exp(2*self.mu*self.h)*(2*K+sigNorm*thetaFunction(self.mu*self.h))
        elif self.compoScheme == "S" and self.odeStepScheme == "IE":
            C = K*(np.exp((2*self.mu-alpha)*self.h)+1) + sigNorm*np.exp((2*self.mu-alpha)*self.h)*thetaFunction(self.mu*self.h)
        else:
            C = 0
    
        upperBound = np.linalg.norm(self.X0,ord=2)**2*np.exp(2*(self.mu-alpha)*t[0,:]) + C*self.h*((np.exp(2*(self.mu-alpha)*t[0,:])-1)/(np.exp(2*(self.mu-alpha)*self.h)-1))
    
    
        return t, X, XmeanSquare, upperBound
    
    
    
# ============== END OF CLASS ===============================================================================


# -- Helper Function -- #
def ife(condition,value,errorMsg,*args):
    if condition: 
        if len(args)>0:
            print(*args)
        return value
    else:
        raise Exception(errorMsg)

def thetaFunction(z):    
    if z == 0.0:
        return 1.0
    else:
        return (1-np.exp(-2*z))/(2*z)
  
    
# --- Sample random variables --- #
def generateRandomVariables(A,sig,h,M,N):  
    print("Computing covariance matrix using trapezoid rule...")
    d = np.shape(A)[0]
    numPoints = 1000
    quadpoints,deltaq = np.linspace(0,h,numPoints+1,retstep=True)    
    C = lambda s: ((la.expm(A*(h-s)))@(sig))@((sig.T)@(la.expm(A*(h-s))).T)
    cov = np.zeros((d,d),dtype=float)
    for k in range(len(quadpoints)-1):
        q = quadpoints[k]
        qp1 = quadpoints[k+1]
        cov += ((C(q)+C(qp1))/2)*deltaq
    print(f"Trace of covariance Matrix: {np.trace(cov)}")    
    print("Generating random variables...")
    np.random.seed(1)
    Z = np.random.multivariate_normal(np.zeros(d,dtype=float),np.identity(d,dtype=float),(M,N)) # standard normal random variables
    if np.linalg.matrix_rank(sig) == d:
        chol = la.cholesky(cov,lower=True)
    else:
        print("Hypoelliptic noise structure: Diffusion matrix is not of full rank.")
        chol = la.cholesky(cov+1e-9*np.identity(d,dtype=float),lower=True)          
    dW = np.sqrt(h)*np.einsum('ij,mnj->mni',sig,Z,optimize=True) # Random variables used for splitting/composition methods
    Y = np.einsum('ij,mnj->mni',chol,Z,optimize=True) # Random variables used for DIEM method
    print("Number of paths sampled:",M)
    return Y,dW





