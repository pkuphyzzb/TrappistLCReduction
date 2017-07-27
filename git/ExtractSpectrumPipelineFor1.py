from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as pl 
import scipy as sp
from scipy import interpolate
from scipy import optimize
from scipy.ndimage.interpolation import shift
from lmfit import Model


#filenamelist=['iddea1ljq', 'idded1p3q', 'idde04myq', 'idde02wdq', 'iddea3hrq', 'idded1o9q', 'iddea3gxq', 'idde03foq', 'idded1pdq', 'idde02xvq', 'idde04lfq', 'idde02w2q', 'iddea1mcq', 'idde01kyq', 'idded1ojq', 'iddea3ikq', 'iddea1liq', 'idded1p2q', 'idde04mxq', 'idde02wcq', 'iddea3hqq', 'idded1o8q', 'iddea3gwq', 'idded1pcq', 'idde02xuq', 'idde02w1q', 'iddea1mbq', 'iddeb1myq', 'iddec1nqq', 'idde01kxq', 'idded1oiq', 'iddea3ijq', 'idded1p1q', 'idde04mwq', 'idde02wbq', 'iddea3hpq', 'idded1o7q', 'iddea3gvq', 'idde03fmq', 'idded1pbq', 'idde02xtq', 'idde02w0q', 'iddea1maq', 'iddeb1mxq', 'iddec1npq', 'idde01kwq', 'idded1ohq', 'iddea3iiq', 'iddea1lgq', 'idded1p0q', 'idde04mvq', 'idde02waq', 'iddea3hoq', 'idded1o6q', 'iddea3guq', 'idde03flq', 'idded1paq', 'idde02xsq', 'iddeb1mwq', 'iddec1noq', 'idde01kvq', 'idded1ogq', 'idde02wyq', 'iddea3ihq', 'idde04muq', 'iddea3hnq', 'idded1o5q', 'iddea3gtq', 'idde03fkq', 'idde02xrq', 'iddea3fzq', 'iddec1nnq', 'idde01kuq', 'idded1ofq', 'idde02wxq', 'iddea3igq', 'iddea1leq', 'idde03f9q', 'idde04mtq', 'iddea3hmq', 'idde04lzq', 'iddea3gsq', 'idde03fjq', 'idde02xqq', 'iddea3fyq', 'iddeb1muq', 'idde01ktq', 'idded1oeq', 'idde02y9q', 'idde02wwq', 'idde03f8q', 'idde04msq', 'idde04lyq', 'idde03fiq', 'idde02xpq', 'iddeb1mtq', 'idded1odq', 'idde02y8q', 'idde02wvq', 'idde03f7q', 'iddec1o4q', 'idde04mrq', 'iddea3gqq', 'idde03fhq', 'idde02xoq', 'iddea3h9q', 'iddeb1msq', 'idde02y7q', 'idde02wuq', 'idde03f6q', 'iddec1o3q', 'idde04mqq', 'idde04lwq', 'idde04n9q', 'iddea3gpq', 'idde03fgq', 'idde02xnq', 'iddea3h8q', 'idde01kqq', 'idded1obq', 'idde02y6q', 'idde02wtq', 'idde03f5q', 'idde04mpq', 'idde02vzq', 'idde04lvq', 'idde04n8q', 'iddea3goq', 'idde03ffq', 'idded1ozq', 'idde02xmq', 'iddea3h7q', 'idde01kpq', 'idded1oaq', 'idde02y5q', 'idde02wsq', 'iddeb1n9q', 'iddec1o1q', 'idde04moq', 'idde02vyq', 'idde04luq', 'idde04n7q', 'iddea3gnq', 'idde03feq', 'idded1oyq', 'idde02xlq', 'iddea3h6q', 'iddec1nhq', 'idde01koq', 'idde02y4q', 'idde02wrq', 'iddeb1n8q', 'iddec1o0q', 'idde04mnq', 'idde02vxq', 'idde04ltq', 'idde04n6q', 'iddea3gmq', 'idded1oxq', 'idde02xkq', 'iddea3h5q', 'idde01knq', 'idde02y3q', 'idde02wqq', 'iddeb1n7q', 'idde04mmq', 'idde02ydq', 'idde04lsq', 'idde04n5q', 'iddea3glq', 'idde03fcq', 'idded1owq', 'idde02xjq', 'iddea3h4q', 'idde01kmq', 'idde04nfq', 'idde02y2q', 'idde04mlq', 'idde02x8q', 'iddea3heq', 'idde02ycq', 'idde04lrq', 'idde04n4q', 'iddea3gkq', 'idde03fbq', 'idded1ovq', 'idde02xiq', 'iddea3h3q', 'idde01klq', 'idde04neq', 'idde02y1q', 'idde02woq', 'iddea3g9q', 'iddeb1n5q', 'idde04mkq', 'idde02x7q', 'iddea3hdq', 'idde02ybq', 'idde04lqq', 'idde04n3q', 'iddea3gjq', 'idde03faq', 'idded1ouq', 'idde02xhq', 'idde04m9q', 'iddea3h2q', 'idde01kkq', 'idde04ndq', 'idde02y0q', 'idde02wnq', 'iddea3g8q', 'iddeb1n4q', 'idde04mjq', 'idde02x6q', 'iddea3hcq', 'idde02yaq', 'idde04lpq', 'idde04n2q', 'iddea3giq', 'idded1otq', 'idde02xgq', 'idde04m8q', 'iddea3h1q', 'idde04ncq', 'idde02wmq', 'iddeb1n3q', 'idde04miq', 'idde02x5q', 'iddea3hbq', 'idde04loq', 'idde04n1q', 'idded1osq', 'idde02xfq', 'iddea3h0q', 'idde04nbq', 'idde02wlq', 'iddea3g6q', 'iddeb1n2q', 'idde02x4q', 'iddea3haq', 'idde03fwq', 'idde04n0q', 'iddea3ggq', 'iddec1nzq', 'idded1orq', 'idde02xeq', 'idde04m6q', 'idde04naq', 'idde02wkq', 'iddea3g5q', 'iddeb1n1q', 'idde01l0q', 'idde04mgq', 'idde02x3q', 'idde03fvq', 'idded1pkq', 'idde04lmq', 'idde02w9q', 'iddea3gfq', 'iddec1nyq', 'idded1oqq', 'idde02xdq', 'idde04m5q', 'idded1p9q', 'idde02wjq', 'iddea3g4q', 'iddea1m8q', 'iddeb1n0q', 'idde04mfq', 'idde02x2q', 'idde03fuq', 'idded1pjq', 'idde04llq', 'idde02w8q', 'iddea3geq', 'iddeb1naq', 'iddec1nxq', 'idded1opq', 'idde02xcq', 'idde04m4q', 'idded1p8q', 'idde02wiq', 'iddea3hwq', 'iddea3g3q', 'iddea1m7q', 'idde04meq', 'idde02x1q', 'idde03ftq', 'idded1piq', 'idde04lkq', 'idde02w7q', 'iddea3gdq', 'iddec1nwq', 'idded1ooq', 'idde04m3q', 'idde02xbq', 'idded1p7q', 'idde02whq', 'iddea3hvq', 'iddea3g2q', 'iddea1m6q', 'idde04mdq', 'idde02x0q', 'idde03fsq', 'idded1phq', 'idde04ljq', 'idde02xzq', 'idde02w6q', 'iddea3gcq', 'idded1onq', 'idde04m2q', 'idde02xaq', 'idded1p6q', 'idde02wgq', 'iddea3g1q', 'iddea1m5q', 'idde04mcq', 'idde03frq', 'idded1pgq', 'idde02xyq', 'idde04liq', 'iddea3gbq', 'iddea1mfq', 'iddec1nuq', 'idde04m1q', 'iddea1llq', 'idde01kcq', 'idded1p5q', 'iddea3htq', 'iddea3g0q', 'idde04mbq', 'iddea3gzq', 'idde03fqq', 'idded1pfq', 'idde02xxq', 'idde04lhq', 'idde02w4q', 'iddea3gaq', 'iddea1meq', 'iddec1ntq', 'idded1olq', 'idde04m0q', 'iddea3imq', 'iddea1lkq', 'idded1p4q', 'idde04mzq', 'idde02weq', 'iddea3hsq', 'idde04maq', 'iddea3gyq', 'idde03fpq', 'idded1peq', 'idde02xwq', 'idde04lgq', 'idde02w3q', 'iddea1mdq', 'iddec1nsq', 'idde01kzq', 'idded1okq', 'iddea3ilq']
#filenamelist=["idde04n2q"]

#filenamelist=[]
dxlist=[]
dylist=[]

wheretoread=open("/Users/apple/Desktop/summerresearch/Apai/Trappist/shift2.txt")
for line in wheretoread.readlines():
	string=line.split(' ')
	dxlist.append(float(string[0]))
	dylist.append(float(string[1]))



def gaussian2d(x, y, amp, x0, y0, sigma_x, sigma_y):
    """
2D Gaussian Function,
X, Y should be from meshgrid
    """
    return amp * np.exp(-(x - x0)**2 / (2 * sigma_x**2) - (y - y0)**2 / (
        2 * sigma_y**2))

def IsCosmicRay(image,error,i,j,size):
	flag = False
	subsize=3
	sigma=error[i][j]
	subcolumn=image[i-min([subsize,abs(i)]):i+min([subsize,abs(size-i)]),:]
	subline=image[:,j-min([subsize,abs(j)]):j+min([subsize,abs(size-j)])]
	if abs(image[i][j]-np.median(subcolumn))>5*sigma and abs(image[i][j]-np.median(subline))>5*sigma:
		flag=True
	return flag


def RemoveBadPixels(image,DQ,error,size,frame):
	for i in range(size):
		for j in range(size):
			if DQ[i][j]&4 or DQ[i][j]&16 or DQ[i][j]&512 or DQ[i][j]&32 or frame[i][j]>1.2 or frame[i][j]<0.8 or not np.isfinite(image[i][j]): #or IsCosmicRay(image,error,i,j,size):
				#print(i)
				#print(j)
				#print(frame[i][j])
				#print(DQ[i][j])
				if i+1<size and j<size-1 and j>=0 and i>=0:
					#newfunc=interpolate.interp2d([i+1,i+1,i+1,i,i,i-1,i-1,i-1],[j+1,j,j-1,j+1,j-1,j+1,j,j-1],[image[i+1][j+1],image[i+1][j],image[i+1][j-1],image[i][j+1],image[i][j-1],image[i-1][j+1],image[i-1][j],image[i-1][j-1]],kind='linear')
					#image[i][j]=newfunc(i,j)
					image[i][j]=1/8*(image[i+1][j+1]+image[i+1][j]+image[i+1][j-1]+image[i][j+1]+image[i][j-1]+image[i-1][j+1]+image[i-1][j]+image[i-1][j-1])
				elif i==size-1 and j<size-1 and j>=0:
					image[i][j]=1/5*(image[i][j+1]+image[i][j-1]+image[i-1][j+1]+image[i-1][j]+image[i-1][j-1])
				elif i==0 and j<size-1 and j>=0:
					image[i][j]=1/5*(image[i][j+1]+image[i][j-1]+image[i+1][j+1]+image[i+1][j]+image[i+1][j-1])
				elif j==size-1 and i<size-1 and i>=0:
					image[i][j]=1/5*(image[i+1][j]+image[i+1][j]+image[i][j-1]+image[i-1][j]+image[i-1][j-1])
				elif j==0 and i<size-1 and i>=0:
					image[i][j]=1/5*(image[i+1][j+1]+image[i+1][j]+image[i][j+1]+image[i-1][j+1]+image[i-1][j])
				elif j==size-1 and i==size-1:
					image[i][j]==1/3*(image[i][j-1]+image[i-1][j-1]+image[i-1][j])
				elif i==0 and j==size-1:
					image[i][j]==1/3*(image[i][j-1]+image[i+1][j-1]+image[i+1][j])
				elif j==0 and i==size-1:
					image[i][j]==1/3*(image[i][j+1]+image[i-1][j+1]+image[i-1][j])
				elif i==0 and j==0:
					image[i][j]=1/3*(image[i][j+1]+image[i+1][j+1]+image[i+1][j])
				#print(image[i][j])
	return image



def fitPeak(im,
            x0,
            y0,
            subSize=10,
            init_params={'amp': 10,
                         'x0': 10,
                         'y0': 10,
                         'sigma_x': 1.0,
                         'sigma_y': 1.0}):
    subImage = im[y0 - subSize:y0 + subSize, x0 - subSize:x0 + subSize]
    peakModel = Model(gaussian2d, independent_vars=['x', 'y'])
    x, y = np.meshgrid(list(range(2 * subSize)), list(range(2 * subSize)))
    p = peakModel.make_params(amp=init_params['amp'],
                              x0=init_params['x0'],
                              y0=init_params['y0'],
                              sigma_x=init_params['sigma_x'],
                              sigma_y=init_params['sigma_y'])
    result = peakModel.fit(subImage,
                           x=x,
                           y=y,
                           params=p, method='powell')
    return (result.values['x0'] + x0 - subSize,
            result.values['y0'] + y0 - subSize)

#
#def GaussianFit(Image,size):
#	pi=3.1415926
#	X=np.linspace(0,size,1)
#	Y=np.linspace(0,size,1)
#	z=Image
#	def residuals(p):
#		[x,y]=np.meshgrid(X,Y)
#		xc,yc,xs,ys,a=p
#		result=np.resize(z-a*np.exp(-0.5*(((x-xc)/xs)**2+((y-yc)/ys)**2)),[1,size*size])
#		return result[0]
#	r=optimize.leastsq(residuals,[size/2,size/2,5,5,100])
#	xc,yc,xs,ys,a=r[0]
#	return xc,yc



def Wavelength_Calibration(dx):
	filetowrite=open("/Users/apple/Desktop/summerresearch/Apai/Trappist/14500data/fitting2.txt",'w')
	a00=8954.31
	a01=9.35925e-2
	a10=4.51423e1
	a11=3.17239e-4
	a12=2.17055e-3
	a13=-7.42504e-7
	a14=3.48639e-7
	a15=3.09213e-7
	hdulist=fits.open("/Users/apple/Desktop/summerresearch/Apai/Trappist/14500data/id4301ouq_ima.fits")
	#hdulist=fits.open("/Users/apple/Desktop/summerresearch/Apai/Trappist/14587data/idde04leq_ima.fits")
	StarImage=hdulist[1].data
	StarError=hdulist[2].data
	StarImage,StarError=mask(StarImage,StarError,266)
	#pl.imshow(StarImage,vmin=0,vmax=100)
	#pl.show()
	pl.imshow(StarImage,vmin=0,vmax=100)
	StarImage=StarImage[130:170,125:165]

	#for line in StarImage:
	#	for element in line:
	#		filetowrite.write(str(element))
	#		filetowrite.write(' ')
	#	filetowrite.write('\n')
	
	pl.show()
	xc,yc=fitPeak(StarImage,25,20,10,{'amp': np.max(StarImage),
                        'x0': 10,
                         'y0': 10,
                         'sigma_x': 1.0,
                         'sigma_y': 1.0})
	print(xc,yc)
	#xc,yc=fitPeak(StarImage,143.5,31,15,{'amp': 17000,
    #                    'x0': 15,
    #                   'y0': 15,
    #                  'sigma_x': 1.0,
    #                 'sigma_y': 1.0})
	#prin#t(xc,yc)
	#yc,xc=25.95,19.05
	xc+=125+374
	yc+=130+374
	print(xc-374,yc-374)
	#yc=143.5+374
	#xc=31+374
	wavelength=[]
	for i in range(1,257):
		Dx=i+486-xc
		wavelength.append(a00+a01*xc+Dx*(a10+a11*xc+a12*yc+a13*xc*xc+a14*xc*yc+a15*yc*yc))
	return wavelength

def mask(image,error,size):
	flag=np.zeros([size,size])+1
	for k in range(5):
		imagep=np.multiply(image,flag)
		med=np.median(imagep)
		i=0
		j=0
		for i in range(size):
			for j in range(size):
				if abs(imagep[i][j]-med)>5*error[i][j]:
					flag[i][j]=0
	med=np.median(imagep)
	imagemasked=(image-med)#*(-flag+1)
	errormasked=error#*(-flag+1)
	return imagemasked, errormasked

def flat_field(dx):
	wavelength=Wavelength_Calibration(dx)
	#image,error=mask(image,error,size)
	file=open("/Users/apple/Desktop/summerresearch/Apai/Trappist/flat_field",'w')
	hdulist=fits.open("/Users/apple/Desktop/summerresearch/Apai/Trappist/WFC3_IR_G141_flat_2.fits")
	coeff1=hdulist[0].data[379:635,379:635]
	coeff2=hdulist[1].data[379:635,379:635]
	coeff3=hdulist[2].data[379:635,379:635]
	coeff4=hdulist[3].data[379:635,379:635]
	frame=np.zeros([256,256])+1;
	for i in range(256):
		for j in range(256):
			x=(wavelength[j]-10600)/6400
			frame[i][j]=coeff1[i][j]+x*coeff2[i][j]+x*x*coeff3[i][j]+x*x*x*coeff4[i][j]
			file.write(str(frame[i][j]))
			file.write(' ')
			if frame[i][j]<0.1:
				frame[i][j]=1
		file.write('\n')
	#print(frame)
	return frame


#print(wav)


filenamelist=["id4301qgq", "id4301paq", "id4301qmq", "id4301pgq", "id4301qsq", "id4301pmq", "id4301qyq", "id4301psq", "id4301pyq", "id4301q5q", "id4301oyq", "id4301p5q", "id4301qfq", "id4301qlq", "id4301pfq", "id4301qrq", "id4301plq", "id4301qxq", "id4301r4q", "id4301pxq", "id4301q4q", "id4301p4q", "id4301qeq", "id4301qkq", "id4301peq", "id4301qqq", "id4301pkq", "id4301qwq", "id4301r3q", "id4301pqq", "id4301pwq", "id4301q3q", "id4301q9q", "id4301p3q", "id4301qdq", "id4301qjq", "id4301pdq", "id4301qpq", "id4301pjq", "id4301qvq", "id4301r2q", "id4301ppq", "id4301pvq", "id4301q2q", "id4301q8q", "id4301p2q", "id4301qcq", "id4301p8q", "id4301qiq", "id4301pcq", "id4301qoq", "id4301piq", "id4301quq", "id4301r1q", "id4301poq", "id4301puq", "id4301q7q", "id4301p1q", "id4301qbq", "id4301p7q", "id4301qhq", "id4301pbq", "id4301qnq", "id4301qtq", "id4301pnq", "id4301r0q", "id4301qzq", "id4301ptq", "id4301q0q", "id4301pzq", "id4301q6q", "id4301p0q", "id4301qaq", "id4301ozq"]

totalflux=[]
totalerror=[]
tflux=[]
terror=[]
time=[]

wav=Wavelength_Calibration(0)
frame=flat_field(0)
print(np.argmin(np.abs(frame)))
print(wav[70:185])

#abnormalfile=open("/Users/apple/Desktop/summerresearch/Apai/Trappist/abnormalimage.txt","w")
for i in range(len(filenamelist)):
	name=filenamelist[i]
	hdulist=fits.open("/Users/apple/Desktop/summerresearch/Apai/Trappist/14500data/%s_ima.fits"%name)
	dx=dxlist[i]
	dy=dylist[i]
	#hdulist=fits.open("/Users/apple/Desktop/summerresearch/Apai/Trappist/14587data/%s_ima.fits"%name)
	#spectrum=[]
	#error=[]
	flatted=[]
	flattede=[]
	trueimage=np.zeros([256,256])
	trueerror=np.zeros([256,256])
	for i in range(5):
		f,fe=hdulist[5*i+1].data[5:261,5:261]/frame,hdulist[5*i+2].data[5:261,5:261]/frame
		#pl.matshow(frame)
		#pl.show()
		f=RemoveBadPixels(f,hdulist[5*i+3].data[5:261,5:261],hdulist[5*i+2].data[5:261,5:261],256,frame)
		fe=RemoveBadPixels(fe,hdulist[5*i+3].data[5:261,5:261],hdulist[5*i+2].data[5:261,5:261],256,frame)
		#print(np.max(f))
		#print(np.min(f))
		#pl.matshow(f)
		#pl.show()
		f,fe=shift(f,[-dx,-dy]),shift(fe,[-dx,-dy])
		#pl.imshow(f)
		#pl.show()
		flatted.append(f)
		flattede.append(fe)
	for i in range(5):
		if i == 4:
			tempspec=flatted[i]
			#spectrum.append(tempspec)
		else:
			tempspec=flatted[i]-flatted[i+1]
			#spectrum.append(tempspec)
		temperror=flattede[i]
		#error.append(temperror)
		imagemasked,errormasked=mask(tempspec,temperror,256)
		trueimage+=imagemasked
		trueerror+=errormasked*errormasked
	image=trueimage
	#pl.imshow(image[150:200,55:200])
	#pl.show()
	#totalflux.append(sum(sum(image)))
	#totalerror.append((np.sqrt(sum(sum(trueerror)))))
	#time.append(hdulist[0].header['EXPEND'])
	#print(sum(sum(image)))
	#if sum(sum(image))<58000000 or sum(sum(image))>61000000:
	#	print(name)
		#for i in range(5):
		#	pl.imshow(flatted[i][100:190,60:220])
		#	print(sum(sum(flatted[i][100:190,60:220])))
		#	pl.show()
	#	abnormalfile.write(name)
	#	abnormalfile.write('\n')
	#pl.matshow(image[130-dx:180-dx,55-dy:210-dy],vmin=0,vmax=2000)
	#pl.show()
	for i in range(12):
		totalflux.append([])
		totalerror.append([])
		totalflux[i].append(sum(sum(image[150:200,75+10*i:85+10*i])))
		totalerror[i].append((np.sqrt(sum(sum(trueerror[150:200,75+10*i:85+10*i])))))
		print(sum(sum(image[150:200,75+10*i:85+10*i])))
	tflux.append(sum(sum(image[150:200,55:220])))
	terror.append(np.sqrt(sum(sum(trueerror[150:200,55:220]))))
	print(sum(sum(image[150:200,55:220])))
	time.append(hdulist[0].header['EXPEND'])
	#if sum(sum(image[130:180,55:210]))<58000000 or sum(sum(image[130-dx:180-dx,55-dy:210-dy]))>60000000:
	#	print(name)
	#	pl.imshow(image)
	#	pl.show()
	#	abnormalfile.write(name)
	#	abnormalfile.write('\n')

flux=[totalflux[i] for i in range(14)]
error=[totalerror[i] for i in range(14)]
time=np.array(time)

index=np.argsort(time)

for j in range(12):
	newflux=[flux[j][index[i]] for i in range(len(filenamelist))]
	newerror=[error[j][index[i]] for i in range(len(filenamelist))]
	error[j]=newerror
	flux[j]=newflux
newtflux=[tflux[index[i]] for i in range(len(filenamelist))]
newterror=[terror[index[i]] for i in range(len(filenamelist))]
newfilename=[filenamelist[index[i]] for i in range(len(filenamelist))]
tflux=newtflux
terror=newterror
filenamelist=newfilename
time=np.sort(time)


for j in range(12):
	savefile=open("/Users/apple/Desktop/summerresearch/Apai/Trappist/Obs1SpectralLightCurve%s.txt"%str(j),'w')
	for i in range(len(time)):
		savefile.write(str(time[i]))
		savefile.write('\t')
		savefile.write(str(flux[j][i]))
		savefile.write('\t')
		savefile.write(str(error[j][i]))
		savefile.write('\t')
		savefile.write(str(filenamelist[i]))
		savefile.write('\t')
		#savefile.write(str(dxlist[i]))
		#savefile.write('\t')
		#savefile.write(str(dylist[i]))
		#savefile.write('\t')
		savefile.write('\n')
	pl.errorbar(time,flux[j],yerr=error[j],fmt='.k')
	pl.show()
	savefile.write(str(wav[80+10*j]))
	savefile.write('\n')

savefile=open("/Users/apple/Desktop/summerresearch/Apai/Trappist/Obs1SpectralLightCurve.txt",'w')
for i in range(len(time)):
	savefile.write(str(time[i]))
	savefile.write('\t')
	savefile.write(str(tflux[i]))
	savefile.write('\t')
	savefile.write(str(terror[i]))
	savefile.write('\t')
	savefile.write('\n')

pl.errorbar(time,tflux,yerr=terror,fmt='.k')
pl.show()
