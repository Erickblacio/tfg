import numpy as np
from numpy import pi,sqrt,exp,cos,sin,arctan,arctan2,arccos,abs,linalg,identity,dot,conj,round
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#Definimos los valores de la conductividad y el rango de frecuencias a T = 0
Real =  np.array([3.18320711e+01, 7.86303412e+00, 2.41325848e+00, 1.12013147e+00,
 6.40258840e-01, 4.13033524e-01, 2.88210124e-01, 2.12465331e-01,
 1.63109404e-01, 1.29182085e-01, 1.04870761e-01, 8.68581386e-02,
 7.31456107e-02, 6.24663149e-02, 5.39879837e-02, 4.71452557e-02,
 4.15432731e-02, 3.68995983e-02, 3.30075173e-02, 2.97135073e-02,
 2.68961001e-02, 2.44748844e-02, 1.88044239e-02, 1.49592052e-02,
 1.22342933e-02, 1.02342253e-02, 8.72830813e-03, 7.56542218e-03,
 6.65152031e-03, 5.92258032e-03, 5.33377394e-03, 4.85659006e-03,
 4.46212992e-03, 4.14271857e-03, 3.87937675e-03, 3.66383723e-03,
 3.49089395e-03, 3.35510255e-03, 3.25483030e-03, 3.18690442e-03,
 3.15288936e-03, 3.15928870e-03, 3.24092631e-03, 3.31022490e-03,
 3.40395866e-03, 3.53490593e-03, 3.70325390e-03, 3.92479205e-03,
 4.21310932e-03, 4.60201925e-03, 5.49370295e-03, 5.89570118e-03,
 7.03304509e-03, 8.97320205e-03, 1.28649233e-02, 2.41432326e-02,
 1.25960957e-01, 2.27735118e-01, 2.39052736e-01, 2.42943911e-01,
 2.44893591e-01, 2.46057659e-01, 2.46828476e-01, 2.47374219e-01,
 2.48926776e-01, 2.49399297e-01, 2.49607086e-01, 2.49712000e-01,
 2.49771994e-01, 2.49803697e-01, 2.49819573e-01, 2.49821276e-01,
 2.49782013e-01, 2.49808640e-01, 2.49832123e-01, 2.49849304e-01,
 2.49865087e-01, 2.49877783e-01, 2.49888525e-01, 2.49898056e-01,
 2.49906189e-01, 2.49913387e-01, 2.49919736e-01, 2.49925391e-01,
 2.49930464e-01, 2.49986045e-01, 2.49992751e-01, 2.49994201e-01,
 2.49995576e-01, 2.49996120e-01, 2.49996321e-01, 2.49996467e-01,
 2.49996529e-01, 2.49996570e-01, 2.49996598e-01, 2.49996605e-01,
 2.49996603e-01, 2.49996592e-01, 2.49996515e-01, 2.49996498e-01,
 2.49996473e-01, 2.49996442e-01, 2.49996404e-01, 2.49996361e-01,
 2.49996311e-01])


Imag =  np.array([ 2.28731512e-21,  1.37263432e+01,  8.42171303e+00,  5.85904110e+00,
  4.46056518e+00,  3.59198474e+00,  3.00269665e+00,  2.57735275e+00,
  2.25611518e+00,  2.00496814e+00,  1.80321889e+00,  1.63756826e+00,
  1.49908793e+00,  1.38155917e+00,  1.28051639e+00,  1.19271050e+00,
  1.11564158e+00,  1.04744693e+00,  9.86644822e-01,  9.32071804e-01,
  8.82795998e-01,  8.38057180e-01,  7.22449063e-01,  6.32025764e-01,
  5.59049320e-01,  4.98653957e-01,  4.47644481e-01,  4.03775103e-01,
  3.65406927e-01,  3.31492144e-01,  3.01102616e-01,  2.73568948e-01,
  2.48366107e-01,  2.25099525e-01,  2.03374159e-01,  1.82927074e-01,
  1.63522728e-01,  1.44882399e-01,  1.26942087e-01,  1.09461738e-01,
  9.22033833e-02,  7.50342160e-02,  4.68958780e-02,  3.58504668e-02,
  2.45194953e-02,  1.28192771e-02,  6.32254135e-04, -1.22162739e-02,
 -2.58567455e-02, -4.06595495e-02, -5.69200494e-02, -7.52875995e-02,
 -9.67318094e-02, -1.23166824e-01, -1.58501452e-01, -2.14568830e-01,
 -3.17168295e-01, -2.21190038e-01, -1.71793891e-01, -1.43054567e-01,
 -1.23334459e-01, -1.08584316e-01, -9.69712662e-02, -8.75028201e-02,
 -5.07471966e-02, -3.36473628e-02, -2.38487583e-02, -1.76749320e-02,
 -1.34220229e-02, -1.03804968e-02, -8.14729536e-03, -6.45815634e-03,
 -5.12232472e-03, -4.07328661e-03, -3.20482820e-03, -2.49945925e-03,
 -1.91310756e-03, -1.41839533e-03, -9.96364026e-04, -6.20415415e-04,
 -3.06296718e-04, -1.06652488e-05,  2.39822425e-04,  4.73951572e-04,
  6.82071022e-04,  4.79732888e-03,  7.77926212e-03,  1.07234669e-02,
  1.36763560e-02,  1.66427302e-02,  1.95817499e-02,  2.25698523e-02,
  2.55586635e-02,  2.85588612e-02,  3.15705913e-02,  3.46402428e-02,
  3.76980584e-02,  4.07927768e-02,  4.39037596e-02,  4.70483789e-02,
  5.02083278e-02,  5.33741151e-02,  5.62806827e-02,  5.97640224e-02,
  6.32989311e-02])


h_w =  np.array([0.00000000e+00, 1.74603175e-03, 3.49206349e-03, 5.23809524e-03,
 6.98412698e-03, 8.73015873e-03, 1.04761905e-02, 1.22222222e-02,
 1.39682540e-02, 1.57142857e-02, 1.74603175e-02, 1.92063492e-02,
 2.09523810e-02, 2.26984127e-02, 2.44444444e-02, 2.61904762e-02,
 2.79365079e-02, 2.96825397e-02, 3.14285714e-02, 3.31746032e-02,
 3.49206349e-02, 3.66666667e-02, 4.20634921e-02, 4.74603175e-02,
 5.28571429e-02, 5.82539683e-02, 6.36507937e-02, 6.90476190e-02,
 7.44444444e-02, 7.98412698e-02, 8.52380952e-02, 9.06349206e-02,
 9.60317460e-02, 1.01428571e-01, 1.06825397e-01, 1.12222222e-01,
 1.17619048e-01, 1.23015873e-01, 1.28412698e-01, 1.33809524e-01,
 1.39206349e-01, 1.44603175e-01, 1.53333333e-01, 1.56666667e-01,
 1.60000000e-01, 1.63333333e-01, 1.66666667e-01, 1.70000000e-01,
 1.73333333e-01, 1.76666667e-01, 1.80000000e-01, 1.83333333e-01,
 1.86666667e-01, 1.90000000e-01, 1.93333333e-01, 1.96666667e-01,
 2.00000000e-01, 2.03333333e-01, 2.06666667e-01, 2.10000000e-01,
 2.13333333e-01, 2.16666667e-01, 2.20000000e-01, 2.23333333e-01,
 2.46031746e-01, 2.68730159e-01, 2.91428571e-01, 3.14126984e-01,
 3.36825397e-01, 3.59523810e-01, 3.82222222e-01, 4.04920635e-01,
 4.27619048e-01, 4.50317460e-01, 4.73015873e-01, 4.95714286e-01,
 5.18412698e-01, 5.41111111e-01, 5.63809524e-01, 5.86507937e-01,
 6.09206349e-01, 6.31904762e-01, 6.54603175e-01, 6.77301587e-01,
 7.00000000e-01, 1.66666667e+00, 2.63333333e+00, 3.60000000e+00,
 4.56666667e+00, 5.53333333e+00, 6.50000000e+00, 7.46666667e+00,
 8.43333333e+00, 9.40000000e+00, 1.03666667e+01, 1.13333333e+01,
 1.23000000e+01, 1.32666667e+01, 1.42333333e+01, 1.52000000e+01,
 1.61666667e+01, 1.71333333e+01, 1.81000000e+01, 1.90666667e+01,
 2.00333333e+01])

#Definimos constantes
h_Gamma = 0.001

def theta(m,d):
    if m >= d:
        return 1
    else:
        return 0

#Definimos la conductividad teorica
def s_xx(hw,m):
    omega = hw + 1j*h_Gamma
    return 1j*(1/pi)*(((m**2)/abs(m))*(1/omega)*theta(m,0) - (omega**2/(2*1j*omega**2))*arctan(1j*omega/(2*abs(m))))

def sig_s_xx(m):
    h_omega = np.array([])

    n_1 = 1000
    int_h_omega_a = 0*m/3
    int_h_omega_b = 10*m/3
    h_omega_1 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_1)

    h_omega = np.append(h_omega, h_omega_1)

    n_2 = 590
    int_h_omega_a = 10*m/3
    int_h_omega_b = 600*m/3
    h_omega_2 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_2)

    h_omega = np.append(h_omega, h_omega_2)

    sigma_sigma_0 = np.array([])

    for i in range(len(h_omega)):
        sigma_sigma_0 = np.append(sigma_sigma_0, s_xx(h_omega[i],m))
    
    return sigma_sigma_0


z = np.zeros([len(h_w)])

mu = 0.1

h_omega = np.array([])

n_1 = 1000
int_h_omega_a = 0*mu/3
int_h_omega_b = 10*mu/3
h_omega_1 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_1)

h_omega = np.append(h_omega, h_omega_1)

n_2 = 590
int_h_omega_a = 10*mu/3
int_h_omega_b = 600*mu/3
h_omega_2 = np.arange(int_h_omega_a,int_h_omega_b,(int_h_omega_b - int_h_omega_a)/n_2)

h_omega = np.append(h_omega, h_omega_2)



#Graficamos los resultados con T = 0
plt.plot(h_w, Real*4,label='Real')
plt.plot(h_w, Imag*4,label='Imaginario')
plt.plot(h_omega, (sig_s_xx(mu)*4).real,label='Real teórico',color = 'pink', alpha=1, linestyle='dotted')
plt.plot(h_omega, (sig_s_xx(mu)*4).imag,label='Imaginario teórico',color = 'lime', alpha=1, linestyle='dotted')
plt.plot(h_w, z, color = 'k',linestyle='dotted')
plt.legend(loc="upper right")
plt.ylabel('sig/sig_0')
plt.xlabel('hw (eV)')
plt.show()

#Definimos los valores de la conductividad y el rango de frecuencias a T = 300
Real_300 =  np.array([3.21727922e+01, 7.94785730e+00, 2.44014918e+00, 1.13348137e+00,
 6.48790773e-01, 4.19453697e-01, 2.93618787e-01, 2.17393053e-01,
 1.67843030e-01, 1.33891489e-01, 1.09662209e-01, 9.18051804e-02,
 7.82979366e-02, 6.78613043e-02, 5.96543445e-02, 5.31059291e-02,
 4.78172247e-02, 4.35029201e-02, 3.99546631e-02, 3.70173722e-02,
 3.45748229e-02, 3.25347018e-02, 2.91105871e-02, 2.68226342e-02,
 2.53375844e-02, 2.44431342e-02, 2.39992199e-02, 2.39112074e-02,
 2.41136254e-02, 2.45607182e-02, 2.52196974e-02, 2.60675557e-02,
 2.70868648e-02, 2.82689162e-02, 3.42325500e-02, 4.23144097e-02,
 5.25090813e-02, 6.48161892e-02, 7.91267635e-02, 9.51548926e-02,
 1.12409528e-01, 1.30247615e-01, 1.47919786e-01, 1.64722557e-01,
 1.80085221e-01, 1.93637077e-01, 2.05219890e-01, 2.14855717e-01,
 2.22693376e-01, 2.28952667e-01, 2.33878824e-01, 2.37711440e-01,
 2.40666769e-01, 2.42930093e-01, 2.44654468e-01, 2.45963112e-01,
 2.46953262e-01, 2.47701259e-01, 2.48265107e-01, 2.48689897e-01,
 2.49009712e-01, 2.49250425e-01, 2.49431561e-01, 2.49567966e-01,
 2.49670707e-01, 2.49748166e-01, 2.49806529e-01, 2.49840606e-01,
 2.49884057e-01, 2.49909362e-01, 2.49928625e-01, 2.49943252e-01,
 2.49954419e-01, 2.49962974e-01, 2.49969554e-01, 2.49996182e-01,
 2.49996337e-01, 2.49996605e-01, 2.49996816e-01, 2.49996837e-01,
 2.49996841e-01, 2.49996836e-01, 2.49996825e-01, 2.49996808e-01,
 2.49996789e-01, 2.49996766e-01, 2.49996739e-01, 2.49996709e-01,
 2.49996671e-01, 2.49996634e-01, 2.49996594e-01, 2.49996549e-01,
 2.49996500e-01, 2.49996447e-01, 2.49996390e-01])


Imag_300 =  np.array([ 3.35489071e-16,  1.38720224e+01,  8.50998155e+00,  5.91919319e+00,
  4.50538556e+00,  3.62676852e+00,  3.03065502e+00,  2.60030107e+00,
  2.27518158e+00,  2.02091677e+00,  1.81663215e+00,  1.64879656e+00,
  1.50845564e+00,  1.38936424e+00,  1.28685865e+00,  1.19777155e+00,
  1.11956731e+00,  1.05033319e+00,  9.88581024e-01,  9.33134384e-01,
  8.83050929e-01,  8.37567000e-01,  7.49149599e-01,  6.76699612e-01,
  6.14776192e-01,  5.61458799e-01,  5.15024342e-01,  4.74114073e-01,
  4.37718932e-01,  4.05061409e-01,  3.75533150e-01,  3.48651211e-01,
  3.24027888e-01,  3.01348298e-01,  2.27735229e-01,  1.70011037e-01,
  1.23006079e-01,  8.39717728e-02,  5.14544359e-02,  2.47030926e-02,
  3.30815178e-03, -1.30513969e-02, -2.47710759e-02, -3.23860648e-02,
 -3.65665307e-02, -3.80525882e-02, -3.75718975e-02, -3.57728977e-02,
 -3.31831983e-02, -3.02027315e-02, -2.71122577e-02, -2.40948801e-02,
 -2.12596097e-02, -1.86628221e-02, -1.63256727e-02, -1.42471384e-02,
 -1.24132597e-02, -1.08025746e-02, -9.39278974e-03, -8.16062325e-03,
 -7.08301404e-03, -6.13932753e-03, -5.31132423e-03, -4.58286770e-03,
 -3.94031350e-03, -3.37319435e-03, -2.86602425e-03, -2.41176366e-03,
 -2.00578273e-03, -1.64050693e-03, -1.31032450e-03, -1.01043684e-03,
 -7.36906663e-04, -4.86373097e-04, -2.55973199e-04,  4.96028842e-03,
  8.04845752e-03,  1.10488904e-02,  1.40385210e-02,  1.70335598e-02,
  2.00392457e-02,  2.30584405e-02,  2.60932660e-02,  2.91454368e-02,
  3.22094189e-02,  3.53617881e-02,  3.84657153e-02,  4.16183396e-02,
  4.48054335e-02,  4.80287435e-02,  5.12909432e-02,  5.45956914e-02,
  5.79465090e-02,  6.13473452e-02,  6.48024956e-02])


hw_300 =  np.array([0.00000000e+00, 1.74603175e-03, 3.49206349e-03, 5.23809524e-03,
 6.98412698e-03, 8.73015873e-03, 1.04761905e-02, 1.22222222e-02,
 1.39682540e-02, 1.57142857e-02, 1.74603175e-02, 1.92063492e-02,
 2.09523810e-02, 2.26984127e-02, 2.44444444e-02, 2.61904762e-02,
 2.79365079e-02, 2.96825397e-02, 3.14285714e-02, 3.31746032e-02,
 3.49206349e-02, 3.66666667e-02, 4.05555556e-02, 4.44444444e-02,
 4.83333333e-02, 5.22222222e-02, 5.61111111e-02, 6.00000000e-02,
 6.38888889e-02, 6.77777778e-02, 7.16666667e-02, 7.55555556e-02,
 7.94444444e-02, 8.33333333e-02, 9.83739837e-02, 1.13414634e-01,
 1.28455285e-01, 1.43495935e-01, 1.58536585e-01, 1.73577236e-01,
 1.88617886e-01, 2.03658537e-01, 2.18699187e-01, 2.33739837e-01,
 2.48780488e-01, 2.63821138e-01, 2.78861789e-01, 2.93902439e-01,
 3.08943089e-01, 3.23983740e-01, 3.39024390e-01, 3.54065041e-01,
 3.69105691e-01, 3.84146341e-01, 3.99186992e-01, 4.14227642e-01,
 4.29268293e-01, 4.44308943e-01, 4.59349593e-01, 4.74390244e-01,
 4.89430894e-01, 5.04471545e-01, 5.19512195e-01, 5.34552846e-01,
 5.49593496e-01, 5.64634146e-01, 5.79674797e-01, 5.94715447e-01,
 6.09756098e-01, 6.24796748e-01, 6.39837398e-01, 6.54878049e-01,
 6.69918699e-01, 6.84959350e-01, 7.00000000e-01, 1.66666667e+00,
 2.63333333e+00, 3.60000000e+00, 4.56666667e+00, 5.53333333e+00,
 6.50000000e+00, 7.46666667e+00, 8.43333333e+00, 9.40000000e+00,
 1.03666667e+01, 1.13333333e+01, 1.23000000e+01, 1.32666667e+01,
 1.42333333e+01, 1.52000000e+01, 1.61666667e+01, 1.71333333e+01,
 1.81000000e+01, 1.90666667e+01, 2.00333333e+01])


z_300 = np.zeros([len(hw_300)])

#Graficamos los resultados con T = 300
plt.plot(hw_300, Real_300*4,label='Real')
plt.plot(hw_300, Imag_300*4,label='Imaginario')
plt.plot(h_omega, (sig_s_xx(mu)*4).real,label='Real teórico',color = 'pink', alpha=1, linestyle='dashed')
plt.plot(h_omega, (sig_s_xx(mu)*4).imag,label='Imaginario teórico',color = 'lime', alpha=1, linestyle='dashed')
plt.plot(hw_300, z_300,label='Cero', color = 'k',linestyle='dotted')
plt.legend(loc="upper right")
plt.ylabel('sig/sig_0')
plt.xlabel('hw (eV)')
plt.show()