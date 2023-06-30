from scipy import integrate
import numpy as np
from numpy import pi,sqrt,exp,cos,sin,arctan,arctan2,arccos,abs
import matplotlib.pyplot as plt
import time
import matplotlib.colors as colors

#a = 1.42

def Kubo(a,vpp_pi,m,T):	#UNIDADES EN eV
	#Definimos constantes
	e = 1
	h_barra = 6.582 * (10**(-16))
	Gamma = 10**(13)
	h_barra_Gamma = 0.001
	T = 10**(-6)
	k_b = 8.617 * 10**(-5)
	c = 3*(10**8)
	eps = 10**-3

	alpha = 1/137
	sigma_0 = alpha * c / 4

	v_f = int(3*abs(vpp_pi)*a/(2*h_barra))
	h_barra_v_f = (3*abs(vpp_pi)*a/2)
	k_max = h_barra_v_f*(4*pi/(3*a))

	#Definimos la matriz velocidad
	def v(i):
		if i == 0:
			return h_barra_v_f*np.array([[0, 1],[1, 0]])
		else:
			return h_barra_v_f*np.array([[0, -1j],[1j, 0]])


	lam = np.array([1,-1])

	#Definimos la energia
	def e_lam_k(i,k):
		return lam[i]*h_barra_v_f*abs(k)

	#Definimos el nivel de Fermi
	def n_f(i,k,m,T):
		if T < (10**-6):
			if (e_lam_k(i,k) - m) > 0:
				return 0
			else:
				return 1
		else:
			return 1/(exp((e_lam_k(i,k) - m)/(k_b*T)) + 1)

	#Definimos el autovector ket
	def u_lam_k_ket(i,k,phi):
		if k == 0:
			return np.array([(-lam[i]*1j)/sqrt(2),1/sqrt(2)])
		else:
			return np.array([-(lam[i]/sqrt(2))*(sin(phi) + 1j*cos(phi)),1/sqrt(2)])

	#Definimos el autovector bra
	def u_lam_k_bra(i,k,phi):
		if k == 0:
			return np.array([(lam[i]*1j)/sqrt(2),1/sqrt(2)])
		else:
			return np.array([-(lam[i]/sqrt(2))*(sin(phi) - 1j*cos(phi)),1/sqrt(2)])

	#Definimos el primer bra-ket
	def bra_ket_1(i_1,i_2,k,phi,q):
		primero = np.array([u_lam_k_bra(i_1,k,phi)[0]*v(0)[0][0] + u_lam_k_bra(i_1,k,phi)[1]*v(0)[1][0],
				u_lam_k_bra(i_1,k,phi)[0]*v(0)[0][1] + u_lam_k_bra(i_1,k,phi)[1]*v(0)[1][1]])

		segundo = primero[0] * u_lam_k_ket(i_2,k + q,phi)[0] + primero[1] * u_lam_k_ket(i_2,k + q,phi)[1]
		return segundo

	#Definimos el segundo bra-ket
	def bra_ket_2(i_1,i_2,k,phi,q):
		primero = np.array([u_lam_k_bra(i_2,k + q,phi)[0]*v(0)[0][0] + u_lam_k_bra(i_2,k + q,phi)[1]*v(0)[1][0],
				u_lam_k_bra(i_2,k + q,phi)[0]*v(0)[0][1] + u_lam_k_bra(i_2,k + q,phi)[1]*v(0)[1][1]])

		segundo = primero[0] * u_lam_k_ket(i_1,k,phi)[0] + primero[1] * u_lam_k_ket(i_1,k,phi)[1]
		return segundo


	def den(i_1,i_2,k,q,h_barra_omega):
		return (h_barra_omega + 1j*h_barra_Gamma + e_lam_k(i_1,k) - e_lam_k(i_2,k + q))


	def f(i_1,i_2,k,phi,q,h_barra_omega):
		return (bra_ket_1(i_1,i_2,k,phi,q)*bra_ket_2(i_1,i_2,k,phi,q))/(h_barra_omega + 1j*h_barra_Gamma + e_lam_k(i_1,k) - e_lam_k(i_2,k + q))


	def f_delta(x,T):
		if T < 10**-6:
			T = 10**-3

		Beta = min(100,1/(k_b*T))
		#return (exp(x*Beta)*Beta) / ((exp(x*Beta) + 1)**2)
		if abs(Beta*x/2) < 25:
			return Beta/(4*(np.cosh(Beta*x/2)**2))
		else:
			return 0


	def division(i_1,i_2,k,q,m,T):
		if (i_1 == i_2) and (abs(q) < eps):
			return -f_delta(e_lam_k(i_1,k) - m,T)

		else:
			return (n_f(i_1,k,m,T) - n_f(i_2,k + q,m,T)) / (e_lam_k(i_1,k) - e_lam_k(i_2,k + q))

	#Definimos la integral
	def integral(h_barra_omega, q,m,T):
		condc = np.array([])
		condc_2 = np.array([])
        
		#Definimos los bucles lambda
		for lam_1 in range(2):
			for lam_2 in range(2):
				func_1 = lambda k, phi: ((1/((2*pi)**2)) * f(lam_1, lam_2, k, phi, q, h_barra_omega) * division(lam_1, lam_2, k, q, m, T) * k).real

				func_2 = lambda k, phi: ((1/((2*pi)**2)) * f(lam_1, lam_2, k, phi, q, h_barra_omega) * division(lam_1, lam_2, k, q, m, T) * k).imag

				int_a_k = 0
				int_b_k = sqrt(((4*pi)/(3*a))**2 + ((4*pi)/(3*a))**2)

				int_a_phi = 0
				int_b_phi = 2*pi

				options_1 = {'limit':50,'epsabs':10**-5}
				kubo_1,err,out_dict = integrate.nquad(func_1,[[int_a_k, int_b_k], [int_a_phi, int_b_phi]], args=(), opts=[options_1,options_1],full_output=True)
				print('Error 1 = ', err, 'N eval 1 = ',out_dict)

				options_2 = {'limit':50,'epsabs':10**-6}
				kubo_2,err,out_dict = integrate.nquad(func_2,[[int_a_k, int_b_k], [int_a_phi, int_b_phi]], args=(), opts=[options_2,options_2],full_output=True)
				print('Error 2 = ', err, 'N eval 2 = ',out_dict)
				print('')
                
				condc = np.append(condc,[(kubo_1)])
				condc_2 = np.append(condc_2,[(kubo_2)])

		return np.sum(condc) + 1j*(np.sum(condc_2))


	#Definimos la grafica de la conducion en funcion de la frecuencia
	def sig_w(m,T):
		#Definimos el rango de frecuencias
		h_omega = np.array([])
		
		n_1 = 21
		int_h_omega_a = 0*m/3
		int_h_omega_b = 1.1*m/3
		h_omega_1 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_1)
		
		h_omega = np.append(h_omega, h_omega_1)
		
		n_2 = 17
		int_h_omega_a = 1.1*m/3
		int_h_omega_b = 2.5*m/3
		h_omega_2 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_2)
		
		h_omega = np.append(h_omega, h_omega_2)
		
		n_3 = 41
		int_h_omega_a = 2.5*m/3
		int_h_omega_b = 21*m/3
		h_omega_3 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_3)
		
		h_omega = np.append(h_omega, h_omega_3)
		
		n_4 = 21
		int_h_omega_a = 21*m/3
		int_h_omega_b = 630*m/3
		h_omega_4 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_4)
		
		h_omega = np.append(h_omega, h_omega_4)

		sigma_r = np.zeros(len(h_omega))
		sigma_i = np.zeros(len(h_omega))
        
		print('hw = ',h_omega)
        
		#Hacemos las integrales
		for i in range(len(h_omega)):
			sigma_sigma_0 = -1*1j * 4 * integral(h_omega[i], (0)*m/(3*h_barra_v_f),m,T)
			sigma_r[i] = (sigma_sigma_0).real
			sigma_i[i] = (sigma_sigma_0).imag
			print('i = ',i)

		#Graficamos los resultados
		print('Real = ',sigma_r)
		print('Imag = ',sigma_i)
		print('hw = ',h_omega)
		plt.plot(h_omega*3/m,sigma_r)
		plt.plot(h_omega*3/m,sigma_i)
		#plt.gca().set_ylim([-0.5,5])
		plt.ylabel('sig/sig_0')
		plt.xlabel('h_w_3/mu')
		plt.show()

	#Definimos la grafica de densidad de la conductividad
	def density(m,T):
		#Definimos el rango de las variables w y q
		n = 31
        
		int_q_a = 0
		int_q_b = 10*m/(3*h_barra_v_f)
        
		q_x_arr = np.linspace(int_q_a, int_q_b, n)
        
		int_h_omega_a = 0
		int_h_omega_b = 10*m/(3)
        
		h_omega = np.linspace(int_h_omega_a, int_h_omega_b, n)
        
		print('hw = ', h_omega*3/m)
		print('q = ', q_x_arr*h_barra_v_f*3/m)
        
		print(len(h_omega),len(q_x_arr))
        
		print('hw = ', h_omega)
		print('q = ', q_x_arr)
		print('q = ', q_x_arr*h_barra_v_f)

		sigma_real = np.zeros((len(q_x_arr),len(h_omega)))
		sigma_imag = np.zeros((len(q_x_arr),len(h_omega)))
		#Hacemos las integrales
		for i in range(len(h_omega)):
			for j in range(len(q_x_arr)):
				sigma_sigma_0 = -1*1j * 4 * integral(h_omega[i],q_x_arr[j],m,T)
				sigma_real[i,j] = (sigma_sigma_0).real
				sigma_imag[i,j] = (sigma_sigma_0).imag
				print('j = ',j)
			print('       i = ',i)
        
		print('hw = ',h_omega)
		print('q = ',q_x_arr)
		print('q = ', q_x_arr*h_barra_v_f)

		print('Real = ',sigma_real)
		print('Imag = ',sigma_imag)
        
		#Graficamos los resultados        
		plt.imshow(sigma_real, interpolation='none', cmap = 'jet',origin = 'lower',extent=[int_q_a*h_barra_v_f*3/(m), int_q_b*h_barra_v_f*3/(m), int_h_omega_a*3/(m), int_h_omega_b*3/(m)], norm=colors.SymLogNorm(linthresh=1.5, linscale=7, vmin=0, vmax=10**4, base=10))
		plt.colorbar()
		plt.ylabel('h_w_3/mu')
		plt.xlabel('h_vf_q_3/mu')
		plt.show()
        
		plt.imshow(sigma_imag, interpolation='none', cmap = 'jet',origin = 'lower',extent=[int_q_a*h_barra_v_f*3/(m), int_q_b*h_barra_v_f*3/(m), int_h_omega_a*3/(m), int_h_omega_b*3/(m)])
		plt.colorbar()
		plt.ylabel('h_w_3/mu')
		plt.xlabel('h_vf_q_3/mu')
		plt.show()



	sig_w(m,T)
	density(m,T)

	h_omega = 0*m/3
	q = 0*m/(3*h_barra_v_f)
	print('Omega = ', h_omega*3/m)
	print('q = ', q*h_barra_v_f*3/m)
    
	return (-1*1j * 4 * integral(h_omega,q,m,T))



start = time.time()
#Definimos las variables para obtener los resultados que buscamos
a = 1.42
v_pp_pi = -2.92

m = 0.1
T = 0

Kubo(a,v_pp_pi,m,T)

m = 0.1
T = 300

Kubo(a,v_pp_pi,m,T)

end = time.time()
print(end - start)
