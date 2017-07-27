import matplotlib.pyplot as pl 
#import quadlimb
import numpy as np
import matplotlib.pyplot as pl
import scipy
import emcee
import corner
import batman
import numpy as np
from matplotlib.ticker import MaxNLocator

filename=open('/Users/apple/Desktop/summerresearch/Apai/Trappist/Raw_Light_Curve.txt')
time=np.array([])
flux=np.array([])
error=np.array([])



for line in filename.readlines():
	string=line.split('\t')
	#print(string)
	time=np.append(time,float(string[0]))
	flux=np.append(flux,float(string[1]))
	error=np.append(error,float(string[2]))
print(len(time))

pl.scatter(time,flux)
pl.show()
######################################CODE_TOMB_FOR_DOUBLE_MODAL_CORRECTION_1############################

ave1=np.mean(flux[0:20:2])*10/19+np.mean(flux[38:49:2])*6/19+np.mean(flux[49:54:2])*3/19
ave2=np.mean(flux[1:19:2])*9/17+np.mean(flux[39:48:2])*5/17+np.mean(flux[50:55:2])*3/17
print(ave2)
print(ave1)
print(ave2/ave1)
flux[0:20:2]*=ave2/ave1
flux[38:49:2]*=ave2/ave1
flux[49:54:2]*=ave2/ave1
flux[20:38:2]*=ave2/ave1

newerror=error
error=error/8

#error=flux*np.sqrt((np.sum((flux[1:19]-np.mean(flux[1:19]))**2)+(np.sum((flux[38:55]-np.mean(flux[38:55]))**2)))/36)/np.mean(flux)
#print(error)
#error=np.sqrt(error)
pl.errorbar(time,flux,yerr=error,fmt='.k')

pl.show()
#filetowrite=open('/Users/apple/Desktop/summerresearch/Apai/Trappist/Raw_light_Curve3.txt','w')
#for t,f,e in zip(time,flux,error):
#	filetowrite.write(str(t))
#	filetowrite.write(' ')
#	filetowrite.write(str(f))
#	filetowrite.write(' ')
#	filetowrite.write(str(e))
#	filetowrite.write(' ')
#	filetowrite.write('\n')
#flux1=np.concatenate(flux[0:20:2],flux[38:49:2])
#flux1=np.concatenate(flux1,flux[49:54:2])
######################################CODE_TOMB_FOR_DOUBLE_MODAL_CORRECTION_1############################

time1=time[0:19]
flux1=flux[0:19]
error1=error[0:19]
time2=time[19:38]
flux2=flux[19:38]
error2=flux[19:38]
time3=time[38:55]
flux3=flux[38:55]
error3=error[38:55]

def lnprior(theta):
	C,R,tau=theta
	if C<59000000 or C>60000000 or R>0.01 or R<0 or tau>0.1 or tau<0:
		return -np.inf

def lnlikeEXP(theta,x,y,yerr):
	C,R,tau=theta
	#if lnprior(theta)==-np.inf:
	#	return -np.inf
	#else:
	model=C*(1-R*np.exp(-(x-x[0])/tau))
	chi2=np.sum(((y-model)/yerr)**2)
	#print(chi2/len(y))
	return -0.5*chi2


#theta0=[1.51,57512,10,90,0.01,2.42,57512,10,90,0.01,0.01,60000000,1,0.01]
theta1=[60000000,0.005,0.04]
theta2=[60000000,0.004,0.05]
pl.figure()
pl.subplot(2,1,1)
pl.scatter(time1,flux1)
pl.plot(time1,theta1[0]*(1-theta1[1]*np.exp(-(time1-time1[0])/theta1[2])))
pl.subplot(2,1,2)
pl.scatter(time3,flux3)
pl.plot(time3,theta2[0]*(1-theta2[1]*np.exp(-(time3-time3[0])/theta2[2])))
pl.show()


print("begin to run the mcmc chain!")

ndim=3
nwalkers=500
pos=[theta1*(1+(np.random.randn(ndim)-0.5)*0.002) for i in range(nwalkers)]

sampler1=emcee.EnsembleSampler(nwalkers,ndim,lnlikeEXP,args=(time1,flux1,error1))

sampler1.run_mcmc(pos,1000)

samples1=sampler1.chain[:,100:,:].reshape((-1,ndim))

C1,R1,tau1=map(lambda v:(v[1],v[2]-v[1],v[1]-v[0]),
	zip(*np.percentile(samples1,[16,50,84],axis=0))) 
print("Fitted C1=%.4f, R1=%.4f, tau1=%.4f"%(C1[0],R1[0],tau1[0]))
print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler1.acceptance_fraction)))
#pl.figure()
#pl.scatter(time,flux1)
#pl.plot(time1,C1[0]*(1-R1[0]*np.exp(-(time1-time1[0])/tau1[0])))
#pl.savefig("oot1.png")
#pl.show()

pos=[theta2*(1+(np.random.randn(ndim)-0.5)*0.002) for i in range(nwalkers)]

sampler2=emcee.EnsembleSampler(nwalkers,ndim,lnlikeEXP,args=(time3,flux3,error3))

sampler2.run_mcmc(pos,1000)

samples2=sampler2.chain[:,100:,:].reshape((-1,ndim))

C2,R2,tau2=map(lambda v:(v[1],v[2]-v[1],v[1]-v[0]),
	zip(*np.percentile(samples2,[16,50,84],axis=0))) 

print("Fitted C2=%.4f, R2=%.4f, tau2=%.4f"%(C2[0],R2[0],tau2[0]))
print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler2.acceptance_fraction)))

pl.figure()
pl.scatter(time,flux)
pl.plot(time1,C1[0]*(1-R1[0]*np.exp(-(time1-time1[0])/tau1[0])))
pl.plot(time3,C2[0]*(1-R2[0]*np.exp(-(time3-time3[0])/tau2[0])))
pl.savefig("/Users/apple/Desktop/summerresearch/Apai/Trappist/RawLightCurvesObs1/Corrected11.png")
pl.show()
print("MCMC chain now done")


#fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
#axes[0].plot(sampler1.chain[:, :, 0].T, color="k", alpha=0.4)
#axes[0].yaxis.set_major_locator(MaxNLocator(5))
#axes[0].set_ylabel("$C1$")

#axes[1].plot(sampler1.chain[:, :, 1].T, color="k", alpha=0.4)
#axes[1].yaxis.set_major_locator(MaxNLocator(5))
#axes[1].set_ylabel("$R1$")

#axes[2].plot(np.exp(sampler1.chain[:, :, 2]).T, color="k", alpha=0.4)
#axes[2].yaxis.set_major_locator(MaxNLocator(5))
#axes[2].set_ylabel("$tau1$")
#pl.show()


#fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
#axes[0].plot(sampler2.chain[:, :, 0].T, color="k", alpha=0.4)
#axes[0].yaxis.set_major_locator(MaxNLocator(5))
#axes[0].set_ylabel("$C2$")

#axes[1].plot(sampler2.chain[:, :, 1].T, color="k", alpha=0.4)
#axes[1].yaxis.set_major_locator(MaxNLocator(5))
#axes[1].set_ylabel("$R2$")

#axes[2].plot(np.exp(sampler2.chain[:, :, 2]).T, color="k", alpha=0.4)
#axes[2].yaxis.set_major_locator(MaxNLocator(5))
#axes[2].set_ylabel("$tau2$")
#pl.show()


#fig1=corner.corner(samples1,labels=["$C1$","$R1$","$tau1$"])
#fig1.savefig("oot1fit1.png")
#fig2=corner.corner(samples2,labels=["$C2$","$R2$","$tau2$"])
#fig2.savefig("oot2fitt.png")


flux1=list(flux1/C1[0]/((1-R1[0]*np.exp(-(time1-time1[0])/tau1[0]))))
flux2=list(flux2/C2[0]/((1-R2[0]*np.exp(-(time2-time2[0])/tau2[0]))))
flux3=list(flux3/C2[0]/((1-R2[0]*np.exp(-(time3-time3[0])/tau2[0]))))


flux1.extend(flux2)
flux1.extend(flux3)
newflux=flux1

error=error*8
error=error/flux*np.array(newflux)
#print(error)

#pl.figure()

#pl.errorbar(time,newflux,yerr=error,fmt='.k')
#pl.show()

filename=open('/Users/apple/Desktop/summerresearch/Apai/Trappist/LighCurveWithSysRemoved11','w')

for i in range(len(time)):
	filename.write(str(time[i]))
	filename.write(' ')
	filename.write(str(newflux[i]))
	filename.write(' ')
	filename.write(str(error[i]))
	filename.write(' ')
	filename.write('\n')






