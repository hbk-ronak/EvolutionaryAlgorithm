import numpy as np
def sphere(x):
	return np.sum(x**2)
	
def parabola(x):
	return (x+4)**2

def ackleys(x):
	a = x[0]
	b = x[1]
	f = -20*np.exp(-0.2*np.sqrt(0.5*(a**2+b**2))) - np.exp(0.5*(np.cos(2*np.pi*a)+np.cos(2*np.pi*b))) + np.exp(1)+20
	return f

def easom(x):
	a = x[0]
	b = x[1]
	f =-np.cos(a)*np.cos(b)*np.exp(-((a-np.pi)**2+(b-np.pi)**2))
	return f

def beales(x):
	a = x[0]
	b = x[1]
	f = (1.5-a+a*b)**2 + (2.25 - a + a*b**2)**2 + (2.625-a+a*b**3)**2
	return f

def goldstein(x):
	a = x[0]
	b = x[1]
	f = (1+(a+b+1)**2*(19-14*a+3*a**2-14*b+6*a*b+3*b**2))* \
	(30+(2*a-3*b)**2*(18-32*a+12*a**2+48*b-36*a*b+27*b**2))
	return f

if __name__ == "__main__":
    pass