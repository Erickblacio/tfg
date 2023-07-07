from scipy import integrate
import numpy as np
from numpy import pi,sqrt,exp,cos,sin,arctan,arctan2,arccos,abs,linalg,identity,dot,conj,round
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#a = 1.42

def Kubo(a,vpp_pi,m,T):		#UNIDADES EN eV
	#Definimos constantes
	e = 1
	h_barra = 6.582 * (10**(-16))
	Gamma = 10**(13)
	h_barra_Gamma = 0.001
	k_b = 8.617 * 10**(-5)
	c = 3*(10**8)
	eps = 10**-3

	alpha = 1/137
	sigma_0 = alpha * c / 4

	v_f = int(3*abs(vpp_pi)*a/(2*h_barra))
	h_barra_v_f = (3*abs(vpp_pi)*a/2)
	k_max = h_barra_v_f*(4*pi/(3*a))

	ep = 0
	es = -2.77
	vi = -2.77
	vs = -12.68

	#Definimos gamma_k
	def gk(kx, ky):
		return (exp(1j*kx*a) + exp(1j * (kx*(-a/2) + ky*(sqrt(3)*a/2))) + exp(1j * (kx*(-a/2) + ky*(-sqrt(3)*a/2))))

	#Definimos gamma_k_transpuesta
	def gkt(kx, ky):
		return (exp(-1j*kx*a) + exp(-1j * (kx*(-a/2) + ky*(sqrt(3)*a/2))) + exp(-1j * (kx*(-a/2) + ky*(-sqrt(3)*a/2))))

	#Definimos el hamiltoniano
	def H(kx, ky):
	    return np.array([[ep, vpp_pi*gk(kx,ky)],
	                    [vpp_pi*gkt(kx,ky), ep]])

	#Definimos la matriz velocidad
	def v(kx, ky):

		return np.array([[0, 1j*1/2*a*vpp_pi*(2*exp(1j*a*kx) - exp(1j*a*((-1/2)*kx - (sqrt(3)/2) * ky)) - exp(1j*a*((-1/2)*kx + (sqrt(3)/2)*ky)))],
						[-1j*1/2*a*vpp_pi*(2*exp(-1j*a*kx) - exp(1j*a*((1/2)*kx + (sqrt(3)/2) * ky)) - exp(1j*a*((1/2)*kx - (sqrt(3)/2)*ky))), 0]])

	def agk(kx, ky):
		return sqrt(3 + 2*cos(a/2*(3*kx - sqrt(3)*ky)) + 2*cos(a/2*(3*kx + sqrt(3)*ky)) + 2*cos(a*sqrt(3)*ky))

	def e_lam_k(i, kx, ky): #ENERGIAS
		if i == 0:
			return ep + abs(vpp_pi)*agk(kx,ky)
		elif i == 1:
			return ep - abs(vpp_pi)*agk(kx,ky)
		else:
			return 1

	#Definimos el nivel de Fermi
	def n_f(i, kx, ky, mu, T):
		if T < (10**-6):
			if (e_lam_k(i,kx,ky) - mu) > 0:
				return 0
			else:
				return 1
		else:
			return 1/(exp((e_lam_k(i,kx,ky) - mu)/(k_b*T)) + 1)

	#Definimos el autovector ket
	def u_lam_k_bra(i, kx, ky):
		w,v = linalg.eig(H(kx,ky))
		return conj(v[:,i])

	#Definimos el autovector bra
	def u_lam_k_ket(i, kx, ky):
		w,v = linalg.eig(H(kx,ky))
		return v[:,i]

	#Definimos el primer bra-ket
	def bra_ket_1(i_1, i_2, kx, ky, qx, qy):
		return dot(dot(u_lam_k_bra(i_1,kx,ky),v(kx,ky)),u_lam_k_ket(i_2,kx + qx,ky + qy))

	#Definimos el segundo bra-ket
	def bra_ket_2(i_1, i_2, kx, ky, qx, qy):
		return dot(dot(u_lam_k_bra(i_2,kx + qx,ky + qy),v(kx,ky)),u_lam_k_ket(i_1,kx,ky))

	def den(i_1, i_2, kx, ky, qx, qy, h_barra_omega):
		return (h_barra_omega + 1j*h_barra_Gamma + e_lam_k(i_1,kx,ky) - e_lam_k(i_2,kx + qx,ky + qy))


	def f(i_1, i_2, kx, ky, qx, qy, h_barra_omega):
		return (bra_ket_1(i_1,i_2,kx,ky,qx,qy)*bra_ket_2(i_1,i_2,kx,ky,qx,qy))/(h_barra_omega + 1j*h_barra_Gamma + e_lam_k(i_1,kx,ky) - e_lam_k(i_2,kx + qx,ky + qy))


	def f_delta(x, T):
		if T < 10**-6:
			T = 10**-3

		Beta = min(100,1/(k_b*T))
		#return (exp(x*Beta)*Beta) / ((exp(x*Beta) + 1)**2)
		if abs(Beta*x/2) < 25:
			return Beta/(4*(np.cosh(Beta*x/2)**2))
		else:
			return 0


	def division(i_1, i_2, kx, ky, qx, qy, mu, T):
		if (i_1 == i_2) and (abs(sqrt(qx**2 + qy**2)) < eps):
			return -f_delta(e_lam_k(i_1,kx,ky) - mu,T)

		else:
			return (n_f(i_1,kx,ky,mu,T) - n_f(i_2,kx + qx,ky + qy,mu,T)) / (e_lam_k(i_1,kx,ky) - e_lam_k(i_2,kx + qx,ky + qy))

	#Definimos la integral
	def integral(h_barra_omega, qx, qy, mu, T):
		condc = np.array([])
		condc_2 = np.array([])

		#Definimos los bucles lambda
		for lam_1 in range(2):
			for lam_2 in range(2):

				func_1 = lambda k_y, k_x: ((1/((2*pi)**2)) * f(lam_1, lam_2, k_x, k_y, qx, qy, h_barra_omega) * division(lam_1, lam_2, k_x, k_y, qx, qy, mu, T)).real

				func_2 = lambda k_y, k_x: ((1/((2*pi)**2)) * f(lam_1, lam_2, k_x, k_y, qx, qy, h_barra_omega) * division(lam_1, lam_2, k_x, k_y, qx, qy, mu, T)).imag

				int_a_kx = 0
				int_b_kx = (4*pi)/(3*a)

				int_a_ky = (-sqrt(3)*pi)/(3*a)
				int_b_ky = (sqrt(3)*pi)/(3*a)


				options_1 = {'limit':50,'epsabs':10**-3}
				kubo_1,err,out_dict = integrate.nquad(func_1,[[int_a_kx, int_b_kx], [int_a_ky, int_b_ky]], args=(), opts=[options_1,options_1],full_output=True)
				print('Error 1 = ', err, 'N eval 1 = ',out_dict)


				options_2 = {'limit':50,'epsabs':10**-4}
				kubo_2,err,out_dict = integrate.nquad(func_2,[[int_a_kx, int_b_kx], [int_a_ky, int_b_ky]], args=(), opts=[options_2,options_2],full_output=True)
				print('Error 2 = ', err, 'N eval 2 = ',out_dict)
				print('')

				condc = np.append(condc,[(kubo_1)])
				condc_2 = np.append(condc_2,[(kubo_2)])


		return np.sum(condc) + 1j*(np.sum(condc_2))


	#Definimos la grafica de la conducion en funcion de la frecuencia
	def sig_w(m,T):
		q = 0.0*m/(3*h_barra_v_f)
		#Definimos el rango de frecuencias
		h_omega = np.array([])
		
		n_1 = 11
		int_h_omega_a = 0*m/3
		int_h_omega_b = 1.1*m/3
		h_omega_1 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_1)
		
		h_omega = np.append(h_omega, h_omega_1)
		
		n_2 = 21
		int_h_omega_a = 1.1*m/3
		int_h_omega_b = 4.5*m/3
		h_omega_2 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_2)
		
		h_omega = np.append(h_omega, h_omega_2)
		
		n_3 = 23
		int_h_omega_a = 4.5*m/3
		int_h_omega_b = 6.7*m/3
		h_omega_3 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_3)
		
		h_omega = np.append(h_omega, h_omega_3)
		
		n_4 = 25
		int_h_omega_a = 6.7*m/3
		int_h_omega_b = 20*m/3
		h_omega_4 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_4)
		
		h_omega = np.append(h_omega, h_omega_4)
		
		n_5 = 39
		int_h_omega_a = 20*m/3
		int_h_omega_b = 58*m/3
		h_omega_5 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_5)
		
		h_omega = np.append(h_omega, h_omega_5)
		
		n_6 = 21
		int_h_omega_a = 58*m/3
		int_h_omega_b = 79*m/3
		h_omega_6 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_6)
		
		h_omega = np.append(h_omega, h_omega_6)
		
		n_7 = 41
		int_h_omega_a = 79*m/3
		int_h_omega_b = 620*m/3
		h_omega_7 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_7)
		
		h_omega = np.append(h_omega, h_omega_7)
        
		sigma_r = np.zeros(len(h_omega))
		sigma_i = np.zeros(len(h_omega))

		print('h_w = ',h_omega*3/m)

		#Hacemos las integrales
		for i in range(len(h_omega)):
			sigma_sigma_0 = -1*1j * 4 * integral(h_omega[i], q, q, m, T)
			sigma_r[i] = (sigma_sigma_0).real
			sigma_i[i] = (sigma_sigma_0).imag
			print('i = ',i)
			print('')

		#Graficamos los resultados
		print('Real = ',sigma_r)
		print('Imag = ',sigma_i)
		print('h_w = ',h_omega*3/m)
		plt.plot(h_omega*3/m,sigma_r)
		plt.plot(h_omega*3/m,sigma_i)
		plt.ylabel('sig/sig_0')
		plt.xlabel('hw_3/mu')
		plt.show()


	sig_w(m,T)

	h_omega = 0*m/3
	q = 0.0*m/(3*h_barra_v_f)
	return (-1*1j * 4 * integral(h_omega,q,q,m,T))




start = time.time()

#Definimos las variables para obtener los resultados que buscamos
a = 1.80
v_pp_pi = -1.15


m = 0.1
T = 0

Kubo(a,v_pp_pi,m,T)

end = time.time()
print(end - start)
