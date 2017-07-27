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

def DoubleTransitModel(theta,x,y,yerr):
	p1,t01,rp1,w1,u1,p2,t02,rp2,w2,u2,V,C=theta
	params1=batman.TransitParams()
	#params1.t0=t01
	params1.t0=57512.3834
	params1.per=1.6377
	params1.rp=rp1
	params1.a=19.5
	params1.inc=90
	params1.ecc=0
	params1.w=w1
	params1.u=[u1]
	params1.limb_dark="linear"
	m1=batman.TransitModel(params1,x)
	model1=m1.light_curve(params1)
	params2=batman.TransitParams()
	params2.t0=57512.3897
	params2.per=2.0198
	params2.rp=rp2
	params2.a=26.7
	params2.inc=90
	params2.ecc=0
	params2.w=w2
	params2.u=[u2]
	params2.limb_dark="linear"
	m2=batman.TransitModel(params2,x)
	model2=m2.light_curve(params2)
	model3=V*(x-x[0])+C
	#model3=-1167273.5778*(x-x[0])+59856592.8761
	#model=(model1+model2-1)*model3
	model=model1+model2-1
	#model=model1
	return model

def lnprob(theta):
	p1,t01,rp1,w1,u1,p2,t02,rp2,w2,u2,V,C=theta
	if t01<57512.3 or t01>57512.5 or t02<57512.3 or t02>57512.5:
		return -np.inf
	#if p1<0 or p2<0 :#p1>10 or p2>10: #or rp1<0 or rp1>1 or rp1<0 or rp2>1 or u1<0 or u2<0 or u1>1 or u2>1 or w1<0 or w2<0:
	#	return -np.inf
	return 0

def lnlike(theta, x, y, yerr):
	if not np.isfinite(lnprob(theta)):
		return -np.inf
	else :
		model=DoubleTransitModel(theta,x,y,yerr)
		chi2=np.sum(((y-model)/yerr)**2)
		#print(chi2/len(x))
		return -0.5*chi2

for number in range(7,12):
	filename=open('/Users/apple/Desktop/summerresearch/Apai/Trappist/LighCurveWithSysRemoved%s.txt'%str(number))
	#filename=open('/Users/apple/Desktop/summerresearch/Apai/Trappist/LighCurveWithSysRemoved.txt')
	time=np.array([])
	flux=np.array([])
	error=np.array([])



	for line in filename.readlines():
		string=line.split(' ')
		#print(string)
		time=np.append(time,float(string[0]))
		flux=np.append(flux,float(string[1]))
		error=np.append(error,float(string[2]))
	print(len(time))

	#pl.scatter(time,flux)
	#pl.show()




	#theta0=[1.51,57512.38,0.085,90,0.2,2.42,57512.4,0.085,90,0.2,0.00001,np.mean(flux[0:19])*19/35+np.mean(flux[39:55])*16/35]
	theta0=[1.6377,57512.3834,0.085,90,0.2,2.0373,57512.3897,0.085,90,0.2,0.00001,np.mean(flux[0:19])*19/35+np.mean(flux[39:55])*16/35]

	p1,t01,rp1,w1,u1,p2,t02,rp2,w2,u2,V,C=theta0

	model=DoubleTransitModel(theta0,time,flux,error)
	chi2perdof=sum(((model-flux)/error)**2)/len(time)
	print(chi2perdof)


	#pl.figure()
	#pl.errorbar(time,flux,yerr=error,fmt='.k')
	#pl.plot(time,model)
	#pl.savefig("oot1.png")
	#pl.show()


	ndim=12
	nwalkers=500
	pos=[theta0*(1+(np.random.randn(ndim)-0.5)*0.0000002) for i in range(nwalkers)]

	sampler=emcee.EnsembleSampler(nwalkers,ndim,lnlike,args=(time,flux,error))

	sampler.run_mcmc(pos,5000)

	samples=sampler.chain[:,100:,:].reshape((-1,ndim))

	p1,t01,rp1,w1,u1,p2,t02,rp2,w2,u2,V,C=map(lambda v:(v[1],v[2]-v[1],v[1]-v[0]),
		zip(*np.percentile(samples,[16,50,84],axis=0))) 
	print("p1=%.4f,t01=%.4f,rp1=%.4f,w1=%.4f,u1=%.4f,p2=%.4f,t02=%.4f,rp2=%.4f,w2=%.4f,u2=%.4f,V=%.4f,C=%.4f"%(p1[0],t01[0],rp1[0],w1[0],u1[0],p2[0],t02[0],rp2[0],w2[0],u2[0],V[0],C[0]))
	print("rp1=%.4f+%.4f-%.4f"%(rp1[0],rp1[1],rp1[2]))
	print("rp2=%.4f+%.4f-%.4f"%(rp2[0],rp2[1],rp2[2]))	
	print("Mean acceptance fraction: {0:.3f}"
                	.format(np.mean(sampler.acceptance_fraction)))

	theta_mcmc=[p1[0],t01[0],rp1[0],w1[0],u1[0],p2[0],t02[0],rp2[0],w2[0],u2[0],V[0],C[0]]

	model=DoubleTransitModel(theta_mcmc,time,flux,error)

	chi2perdof=sum(((model-flux)/error)**2)/len(time)
	print(chi2perdof)


	pl.figure()
	pl.subplot(2,1,1)
	pl.scatter(time,flux)
	pl.plot(time,model)
	pl.subplot(2,1,2)
	pl.scatter(time,flux-model)
#pl.savefig("oot1.png")
	pl.savefig('FitResult%s.png'%str(number))
	pl.show()
	print("Fig %s has been plotted"%str(number))
'''
fig=corner.corner(samples,labels=["$p1$","$t01$","$rp1$","$w1$","$u11$","p2","t02","rp2","w2","u12","V","C"])
fig.savefig("DoubleTransitTriangle.png")

fig, axes = pl.subplots(12, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("$p1$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("$t01$")

axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("$rp1$")

axes[3].plot(np.exp(sampler.chain[:, :, 3]).T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("$w1$")

axes[4].plot(np.exp(sampler.chain[:, :, 4]).T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].set_ylabel("$u11$")

axes[5].plot(np.exp(sampler.chain[:, :, 5]).T, color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].set_ylabel("$p2$")

axes[6].plot(np.exp(sampler.chain[:, :, 6]).T, color="k", alpha=0.4)
axes[6].yaxis.set_major_locator(MaxNLocator(5))
axes[6].set_ylabel("$t02$")

axes[7].plot(np.exp(sampler.chain[:, :, 7]).T, color="k", alpha=0.4)
axes[7].yaxis.set_major_locator(MaxNLocator(5))
axes[7].set_ylabel("$rp2$")

axes[8].plot(np.exp(sampler.chain[:, :, 8]).T, color="k", alpha=0.4)
axes[8].yaxis.set_major_locator(MaxNLocator(5))
axes[8].set_ylabel("$w2$")

axes[9].plot(np.exp(sampler.chain[:, :, 9]).T, color="k", alpha=0.4)
axes[9].yaxis.set_major_locator(MaxNLocator(5))
axes[9].set_ylabel("$u12$")

axes[10].plot(np.exp(sampler.chain[:, :, 10]).T, color="k", alpha=0.4)
axes[10].yaxis.set_major_locator(MaxNLocator(5))
axes[10].set_ylabel("$V$")

axes[11].plot(np.exp(sampler.chain[:, :, 11]).T, color="k", alpha=0.4)
axes[11].yaxis.set_major_locator(MaxNLocator(5))
axes[11].set_ylabel("$C$")
axes[11].set_xlabel("step number")
print('fig1 picutred!')
fig.savefig("chainsrunning.png")
'''