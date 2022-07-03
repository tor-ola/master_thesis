import numpy as np 
import matplotlib.pyplot as plt


hlist = np.array([1.0,0.1,0.01])
boxRange = 10
alphaList = np.linspace(-100,100,10000)

fig1 = plt.figure(1)
ax1 = plt.subplot(111)

ax1.plot(alphaList,alphaList,color="black",label="$\\alpha$")
ax1.set_ylim(-boxRange,boxRange)
ax1.set_xlim(-boxRange,boxRange)
redColor = np.array([242.0, 17.0, 17.0])/255
blueColor = color= np.array([0.0,137.0,255.0])/255
greenColor=np.array([15.0,205.0, 53.0])/255
purpleColor=np.array([177.0,13.0,236.0])/255
goldColor = np.array([204.0,209.0,53.0])/255
orangeColor = np.array([238.0,156.0,16.0])/255
colorList = (greenColor,blueColor,redColor)
k = 0

for h in hlist:
    numAlphaLowerBound = -1/(2*h)+0.0000000000001/(2*h)
    numAlphaValues = np.linspace(numAlphaLowerBound,100,10000)
    numAlphaList = 1/(2*h)*np.log(1+2*numAlphaValues*h)
    plt.plot(numAlphaValues,numAlphaList, label=f"$\\tilde\\alpha_0, h_0={h})$",color=colorList[k])
    k += 1
    
ax1.set_xlim(-boxRange,boxRange)
ax1.set_ylim(-boxRange,boxRange)
ax1.set_xlabel("$\\alpha$",fontsize=20)
ax1.tick_params(axis="both", which="major", labelsize=15)
ax1.set_title("$\\tilde\\alpha_0$ vs $\\alpha$", fontsize=25)
box1 = ax1.get_position()
ax1.set_position([box1.x0+0.02, box1.y0+0.1, box1.width, box1.height*0.9])
ax1.legend(loc="upper left",fontsize=15,edgecolor="black")


plt.show()





