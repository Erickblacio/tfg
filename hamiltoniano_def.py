from cubature import cubature
import numpy as np
from numpy import pi,sqrt,exp,cos,sin,arctan,arctan2,arccos,abs,conj
import sympy as sy
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import time
'''
DEFINIMOS CONTANTES
'''
Es = -8.31
Ep = 0
A = 0
a = 1.42
a0 = 0.529

e_pi = A**2*Es + (1-A**2)*Ep
e_sigma = ((1 - (A**2))/3) * Es + ((2 + (A**2))/3) * Ep

Zs = 4.84
Zpx = 5.49
Zpz = 4.03
#Definimos los vectores de los atomos vecinos
d1 = np.array([a,0])
d2 = np.array([-a/2,(sqrt(3)*a)/2])
d3 = np.array([-a/2,(-sqrt(3)*a)/2])
'''
PASAMOS DE COORDENADAS CARTESIANAS A POLARES
'''
def r(x,y,z):
	return sqrt(x**2 + y**2 + z**2)

def theta(x,y,z):
	if z == 0:
		return pi/2
	else:
		return arccos(z/r(x,y,z))

def phi(x,y,z):
	return arctan2(y,x)
'''
DEFINIMOS ORBITALES |S> Y |P>
'''
def s(x,y,z):
	return (1/(2*sqrt(2))) * ((Zs/a0)**(3/2)) * (2-(Zs*r(x,y,z))/a0) * (exp(-(Zs*r(x,y,z))/(2*a0))) * (1/sqrt(4*pi))

def pz(x,y,z):
    return sqrt(1/24) * sqrt((Zpz/a0)**3) * (Zpz*r(x,y,z)/a0) * exp(-(Zpz*r(x,y,z))/(2*a0)) * sqrt(3/(4*pi)) * cos(theta(x,y,z))

def px(x,y,z):
    return sqrt(1/24) * sqrt((Zpx/a0)**3) * (Zpx*r(x,y,z)/a0) * exp(-(Zpx*r(x,y,z))/(2*a0)) * sqrt(3/(4*pi)) * sin(theta(x,y,z)) * cos(phi(x,y,z))

def py(x,y,z):
    return sqrt(1/24) * sqrt((Zpx/a0)**3) * (Zpx*r(x,y,z)/a0) * exp(-(Zpx*r(x,y,z))/(2*a0)) * sqrt(3/(4*pi)) * sin(theta(x,y,z)) * sin(phi(x,y,z))

'''
DEFINIMOS LOS ORBITALES DEL CARBONO
'''
def Orb0(x,y,z):
	return A*s(x,y,z) + sqrt(1 - (A**2))*pz(x,y,z)

def Orb1(x,y,z):
	return sqrt((1 - (A**2))/3)*s(x,y,z) + sqrt(2/3)*px(x,y,z) - (A/sqrt(3))*pz(x,y,z)

def Orb2(x,y,z):
	return sqrt((1 - (A**2))/3)*s(x,y,z) - sqrt(1/6)*px(x,y,z) - sqrt(1/2)*py(x,y,z) - (A/sqrt(3))*pz(x,y,z)

def Orb3(x,y,z):
	return sqrt((1 - (A**2))/3)*s(x,y,z) - sqrt(1/6)*px(x,y,z) + sqrt(1/2)*py(x,y,z) - (A/sqrt(3))*pz(x,y,z)

'''
DEFINIMOS LOS ORBITALES CONJUGADOS
'''
def Orb0_c(x,y,z):
	return A*s(x,y,z) + sqrt(1 - (A**2))*pz(x,y,z)

def Orb1_c(x,y,z):
	return sqrt((1 - (A**2))/3)*s(x,y,z) + sqrt(2/3)*conj(px(x,y,z)) - (A/sqrt(3))*pz(x,y,z)

def Orb2_c(x,y,z):
	return sqrt((1 - (A**2))/3)*s(x,y,z) - sqrt(1/6)*conj(px(x,y,z)) - sqrt(1/2)*conj(py(x,y,z)) - (A/sqrt(3))*pz(x,y,z)

def Orb3_c(x,y,z):
	return sqrt((1 - (A**2))/3)*s(x,y,z) - sqrt(1/6)*conj(px(x,y,z)) + sqrt(1/2)*conj(py(x,y,z)) - (A/sqrt(3))*pz(x,y,z)

'''
DEFINIMOS LA ENERGIA DE SALTO PARA SER INTEGRADAS
'''
def Ene_n():
	m = 9.109*(10**(-31))
	h = 6.582*(10**(-16))
	K = 14.3996
	Z = 6
	n = 2
	e = 1

	return -(m/(2*(h**2))) * (K*Z*(e**2)) * 1/(n**2)

def t00(x,y,z):#V_pp_pi
	return (Orb0_c(x-d1[0],y-d1[1],z)*Ene_n()*Orb0(x,y,z))

'''
INTEGRALES 3D V_pp_pi
'''
def integral_t00():
	def func_t00(x_array, *args):
		x,y,z = x_array
		return np.array([t00(x,y,z)])

	n = 8
	int_a = -n*a0
	int_b = n*a0
    #Establecemos el rengo de integracion
	xmin = np.array([int_a,int_a,int_a])
	xmax = np.array([int_b,int_b,int_b])
	ndim = 3
    #Integral
	Vpp,err = cubature(func_t00,ndim,1,xmin,xmax,abserr = 0.001)
	print('V_pppi = ')
	print(Vpp,err)
	return Vpp

'''
ALARGAMIENTO DE LA DISTANCIA DE ENLACE a = [1.42, 2.00] E INTEGRALES
'''
def integrales_vpp_pi():
	def vpppi(x,y,z,i):
		return (Orb0(x-i,y,z)*Ene_n()*Orb0(x,y,z))

	Arr = np.array([])
	n = np.arange(1.42,2,0.02)
    #Bucle de 1.42 a 2.00 saltando de 0.02 en 0.02
    
	for i in n:
		def func(x_array, *args):
			x,y,z = x_array
			return np.array([vpppi(x,y,z,i)])

		n = 8
		int_a = -n*a0
		int_b = n*a0
        #Establecemos el rengo de integracion
		xmin = np.array([int_a,int_a,int_a])
		xmax = np.array([int_b,int_b,int_b])
		ndim = 3
        #Integral
		I,err = cubature(func,ndim,1,xmin,xmax,abserr = 0.001)

		Arr = np.append(Arr,[I])

	t = np.arange(1.42,2,0.02)

    #Representamos los resultados en una grafica
	plt.plot(t,Arr)

	plt.ylabel('E')
	plt.xlabel('ax')
	plt.show()

	n = 10
	int_a = -n*a0
	int_b = n*a0

	return Arr


'''
GRAFICAS DE LAS BANDAS PI CON A = 0
'''
def grafica_E_pi(vpp_pi,q):
	n = 10
	int_a = -n*a0
	int_b = n*a0
	#Redefinimos los vectores al vecino mas cercano para variar la distancia a_x
	d_1 = np.array([q,0])
	d_2 = np.array([-q/2,(sqrt(3)*a)/2])
	d_3 = np.array([-q/2,(-sqrt(3)*a)/2])
    
    #Definimos las bandas pi
	def gamma_k(x,y):
		k = np.array([x,y])
		return sqrt(3 + 2*cos(np.dot(k,(d_1 - d_2))) + 2*cos(np.dot(k,(d_1 - d_3))) + 2*cos(np.dot(k,(d_2 - d_3))))

	def E_pi_pos(x,y):
		return e_pi + abs(vpp_pi) * gamma_k(x,y)

	def E_pi_neg(x,y):
		return e_pi - abs(vpp_pi) * gamma_k(x,y)
    
    #Representamos las bandas en una grafica
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = y = np.arange(-2.5, 2.5, 0.05)
	X, Y = np.meshgrid(x, y)
	zs = np.array([E_pi_pos(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z1 = zs.reshape(X.shape)
	zs = np.array([E_pi_neg(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z2 = zs.reshape(X.shape)

	ax.plot_surface(X, Y, Z1)
	ax.plot_surface(X, Y, Z2)

	ax.set_xlabel('kx')
	ax.set_ylabel('ky')
	ax.set_zlabel('E')
	ax.set(zlim=(-10,10))
	plt.show()

'''
GRAFICAS DE LAS BANDAS SIGMA CON A = 0
'''
def grafica_E_sigma():
	n = 10
	int_a = -n*a0
	int_b = n*a0
    #Definimos constantes
	Vss_sigma = -5
	Vpp_sigma = 8.4
	Vsp_sigma = 5.4
    #Calculamos v_sigma y v_int
	V_sigma = -(2/3)*Vpp_sigma + ((1-A**2)/3)*Vss_sigma - (2/3)*sqrt(2*(1-A**2))*Vsp_sigma
	Vintra = ((1-A**2)/3)*(Es-Ep)

	print('Vsigma = ',V_sigma,' Vint = ',Vintra)
    #Definimos las bandas sigma
	def gamma_k(x,y):
		k = np.array([x,y])
		return sqrt(3 + 2*cos(np.dot(k,(d1 - d2))) + 2*cos(np.dot(k,(d1 - d3))) + 2*cos(np.dot(k,(d2 - d3))))

	def E_sigma_1_pos(x,y):
		return e_sigma - Vintra + V_sigma

	def E_sigma_1_neg(x,y):
		return e_sigma - Vintra - V_sigma

	def E_sigma_2_pos(x,y):
		return e_sigma + Vintra/2 + sqrt((3*Vintra/2)**2 + V_sigma**2 + abs(Vintra*V_sigma)*gamma_k(x,y))

	def E_sigma_2_neg(x,y):
		return e_sigma + Vintra/2 + sqrt((3*Vintra/2)**2 + V_sigma**2 - abs(Vintra*V_sigma)*gamma_k(x,y))

	def E_sigma_3_pos(x,y):
		return e_sigma + Vintra/2 - sqrt((3*Vintra/2)**2 + V_sigma**2 + abs(Vintra*V_sigma)*gamma_k(x,y))

	def E_sigma_3_neg(x,y):
		return e_sigma + Vintra/2 - sqrt((3*Vintra/2)**2 + V_sigma**2 - abs(Vintra*V_sigma)*gamma_k(x,y))
    
    #Representamos las bandas en una grafica
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = y = np.arange(-2.5, 2.5, 0.05)
	X, Y = np.meshgrid(x, y)
	zs = np.array([E_sigma_1_pos(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z1 = zs.reshape(X.shape)
	zs = np.array([E_sigma_1_neg(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z2 = zs.reshape(X.shape)
	zs = np.array([E_sigma_2_pos(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z3 = zs.reshape(X.shape)
	zs = np.array([E_sigma_2_neg(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z4 = zs.reshape(X.shape)
	zs = np.array([E_sigma_3_pos(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z5 = zs.reshape(X.shape)
	zs = np.array([E_sigma_3_neg(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z6 = zs.reshape(X.shape)

	ax.plot_surface(X, Y, Z1)
	ax.plot_surface(X, Y, Z2)

	ax.plot_surface(X, Y, Z3)
	ax.plot_surface(X, Y, Z4)

	ax.plot_surface(X, Y, Z5)
	ax.plot_surface(X, Y, Z6)

	ax.set_xlabel('kx')
	ax.set_ylabel('ky')
	ax.set_zlabel('E')

	plt.show()

'''
GRAFICA DEL CONO DE DIRAC
'''
def grafica_cono_dirac(Vpp_pi):
	int_a = -0.01
	int_b = 0.01
    #Definimos el cono de dirac
	def e_pi_pos(x,y):
		return 3/2*abs(Vpp_pi)*a*sqrt(x**2+y**2)

	def e_pi_neg(x,y):
		return -3/2*abs(Vpp_pi)*a*sqrt(x**2+y**2)
    
    #Representamos las bandas en una grafica

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = y = np.arange(int_a, int_b, 0.0005)
	X, Y = np.meshgrid(x, y)
	zs = np.array([e_pi_pos(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z1 = zs.reshape(X.shape)
	zs = np.array([e_pi_neg(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
	Z2 = zs.reshape(X.shape)

	ax.plot_surface(X, Y, Z1)
	ax.plot_surface(X, Y, Z2)

	ax.set_xlabel('kx')
	ax.set_ylabel('ky')
	ax.set_zlabel('E')

	plt.show()

'''
GRAFICA DE K,T,M,K

Vamos de K = 2pi/3a(1,1/sqrt(3))
a T = (0,0)
a M = (sqrt(3)*a/2,0)
a K otra vez

Intervalo = (K-T)/1000
'''
def grafica_plano_energias(Vpp_pi):
	a = 1.42
    #Definimos constantes
	Vss_sigma = -5
	Vpp_sigma = 8.4
	Vsp_sigma = 5.4
    #Calculamos v_sigma y v_int
	V_sigma = -(2/3)*Vpp_sigma + ((1-A**2)/3)*Vss_sigma - (2/3)*sqrt(2*(1-A**2))*Vsp_sigma
	Vintra = ((1-A**2)/3)*(Es-Ep)
    
    #Definimos los puntos K,Gamma y M
	K = np.array([2*pi/(3*a),2*pi/(sqrt(3)*3*a)])

	Gamma = np.array([0,0])

	M = np.array([2*pi/(3*a),0])
    
    #Definimos las bandas sigma

	def gamma_k(x,y):
		k = np.array([x,y])
		return sqrt(3 + 2*cos(np.dot(k,(d1 - d2))) + 2*cos(np.dot(k,(d1 - d3))) + 2*cos(np.dot(k,(d2 - d3))))

	def E_pi_pos(x,y):
		return e_pi + abs(Vpp_pi) * gamma_k(x,y)

	def E_pi_neg(x,y):
		return e_pi - abs(Vpp_pi) * gamma_k(x,y)

	def E_sigma_1_pos(x,y):
		return e_sigma - Vintra + V_sigma

	def E_sigma_1_neg(x,y):
		return e_sigma - Vintra - V_sigma

	def E_sigma_2_pos(x,y):
		return e_sigma + Vintra/2 + sqrt((3*Vintra/2)**2 + V_sigma**2 + abs(Vintra*V_sigma)*gamma_k(x,y))

	def E_sigma_2_neg(x,y):
		return e_sigma + Vintra/2 + sqrt((3*Vintra/2)**2 + V_sigma**2 - abs(Vintra*V_sigma)*gamma_k(x,y))

	def E_sigma_3_pos(x,y):
		return e_sigma + Vintra/2 - sqrt((3*Vintra/2)**2 + V_sigma**2 + abs(Vintra*V_sigma)*gamma_k(x,y))

	def E_sigma_3_neg(x,y):
		return e_sigma + Vintra/2 - sqrt((3*Vintra/2)**2 + V_sigma**2 - abs(Vintra*V_sigma)*gamma_k(x,y))

	T1 = np.array([])
	T2 = np.array([])
	T3 = np.array([])
	T4 = np.array([])
	T5 = np.array([])
	T6 = np.array([])
	T7 = np.array([])
	T8 = np.array([])
    
    #Definimos el recorrido del corte yendo de K a Gamma
	Intervalo_1_x = (Gamma[0]-K[0])/100
	Intervalo_1_y = (Gamma[1]-K[1])/100

	x_r = np.arange(K[0], Gamma[0], Intervalo_1_x)
	y_r = np.arange(K[1], Gamma[1], Intervalo_1_y)

	for i in range(100):
		T1 = np.append(T1,[E_pi_pos(x_r[i],y_r[i])])
		T2 = np.append(T2,[E_pi_neg(x_r[i],y_r[i])])
		T3 = np.append(T3,[E_sigma_1_pos(x_r[i],y_r[i])])
		T4 = np.append(T4,[E_sigma_1_neg(x_r[i],y_r[i])])
		T5 = np.append(T5,[E_sigma_2_pos(x_r[i],y_r[i])])
		T6 = np.append(T6,[E_sigma_2_neg(x_r[i],y_r[i])])
		T7 = np.append(T7,[E_sigma_3_pos(x_r[i],y_r[i])])
		T8 = np.append(T8,[E_sigma_3_neg(x_r[i],y_r[i])])
    
    #Definimos el recorrido del corte yendo de Gamma a M

	Intervalo_2_x = (M[0]-Gamma[0])/100
	Intervalo_2_y = (M[1]-Gamma[1])/100

	x_r_2 = np.arange(Gamma[0], M[0], Intervalo_2_x)

	for i in range(100):
		T1 = np.append(T1,[E_pi_pos(x_r_2[i],0)])
		T2 = np.append(T2,[E_pi_neg(x_r_2[i],0)])
		T3 = np.append(T3,[E_sigma_1_pos(x_r_2[i],0)])
		T4 = np.append(T4,[E_sigma_1_neg(x_r_2[i],0)])
		T5 = np.append(T5,[E_sigma_2_pos(x_r_2[i],0)])
		T6 = np.append(T6,[E_sigma_2_neg(x_r_2[i],0)])
		T7 = np.append(T7,[E_sigma_3_pos(x_r_2[i],0)])
		T8 = np.append(T8,[E_sigma_3_neg(x_r_2[i],0)])
    
    #Definimos el recorrido del corte yendo de M a K

	Intervalo_3_y = (K[1]-M[1])/100

	y_r_3 = np.arange(M[1], K[1], Intervalo_3_y)

	for i in range(100):
		T1 = np.append(T1,[E_pi_pos(M[0],y_r_3[i])])
		T2 = np.append(T2,[E_pi_neg(M[0],y_r_3[i])])
		T3 = np.append(T3,[E_sigma_1_pos(M[0],y_r_3[i])])
		T4 = np.append(T4,[E_sigma_1_neg(M[0],y_r_3[i])])
		T5 = np.append(T5,[E_sigma_2_pos(M[0],y_r_3[i])])
		T6 = np.append(T6,[E_sigma_2_neg(M[0],y_r_3[i])])
		T7 = np.append(T7,[E_sigma_3_pos(M[0],y_r_3[i])])
		T8 = np.append(T8,[E_sigma_3_neg(M[0],y_r_3[i])])

	x = np.arange(0,3,0.01)
    
    #Representamos el corte de las bandas en una grafica

	plt.plot(x, T1)
	plt.plot(x, T2)
	plt.plot(x, T3)
	plt.plot(x, T4)
	plt.plot(x, T5)
	plt.plot(x, T6)
	plt.plot(x, T7)
	plt.plot(x, T8)

	plt.ylabel('E')

	plt.show()


print('INTEGRALES 3D')
integral_t00()
print('')

#Variación de las energias de salto V_pp_pi
vpp_pi_s = integrales_vpp_pi()

#Grafica bandas pi con a = 1.42
grafica_E_pi(vpp_pi_s[0],1.42)

#Grafica bandas pi con a = 2.00
grafica_E_pi(vpp_pi_s[len(vpp_pi_s) - 1],2.00)

#Grafica cono de Dirac
grafica_cono_dirac(vpp_pi_s[0])

#Grafica bandas sigma
grafica_E_sigma()

#Grafica entre las bandas
grafica_plano_energias(vpp_pi_s[0])
