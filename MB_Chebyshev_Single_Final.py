#This method is not stable when we change the collocation point numbers
#import pybamm
import numpy as np
#from MB_Boundary import Boundary
import matplotlib.pyplot as plt
from MB_Time_Current_Boundary import Time_Current_Boundary

from scipy.integrate import odeint

import os
os.system('cls' if os.name == 'nt' else 'clear')   # To clear terminal

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

rtol = 1e-5 #1e-7
atol = 1e-8 #1e-8

N_collocation = 6
Np = N_collocation - 1 

dt= 1
tfinal= 3600 *6         # 360*5 ; \With small values batery may not be chsarged

Td = 580            #  580 ; Diffusion time constant
capacity= 4.9 * 3600   # 4.9 [A.h] ==> [4.9* 3600 A.sec]
Amplitude = 10
#SoC initial--------------------------------------------------------------
Init = 0  # Initial condition of SoC only at collocation points, not in the center and surface 

Z1= np.ones(Np-1) * Init
Z = np.ones(Np+1) * Init

SoC_Surface= Z[Np]

current=1
surface_flux = current*Td/capacity

Z0=Z.copy()

if tfinal<dt:
   tfinal=100*dt

if dt>=1:
  t = np.linspace(0, tfinal, tfinal)  # Create an array of 100 time points between 0 and t4
else:
  t = np.arange(0, tfinal + dt, dt)

def Xn_D_D2(r_center=0, r_surface=1, P = Np):
  Sn=np.zeros(P+1)
  Xn=np.zeros(P+1)
  D = np.zeros((P+1, P+1))
  D2= np.zeros((P+1, P+1))
  D2= np.zeros((P+1, P+1))  # Create a NumPy array for D2
  
  Xn[0]=1  
  for j in range (0,P+1):
    Sn[j] = -(np.cos(j*np.pi/Np))
    Xn[j] = 0.5*(1+ np.cos(j*np.pi/Np))
  
  for i in range (0,P+1):
    for j in range (0,P+1):  
      if (i==j and i==0):
         D[i][j]= -(2*P**2+1)/6
      elif (i==j and i==P):
         D[i][j]= (2*P**2+1)/6
      elif (i==j and i<=P-1):
         D[i][j]=-Sn[i]/(2*(1-Sn[i]**2))
      else:
         if i==0 or i==P: 
            ci=2
         else:
            ci=1

         if j==0 or j==P: 
            cj=2 
         else:
            cj=1       
         
         D[i][j] = (ci/cj*(-1)**(i+j)/(Sn[i]-Sn[j])) 
  
  print("________________________________________________________")    
  print("Number of Collocation points:", Np)
  #print("Sn (Collocation points):", Sn)
  #print("_______Derivative Matrix: _________________________________________________")    
  D=D*2
  #print("D:", D)
  #D2=multiply_matrices(D, D)  
  
  D2 = D @ D
  #print("______Second Derivative Matrix:__________________________________________________________________________________________________________")    
  #print("D2:",D2)

  # Transform collocation points to actual radius
  r_vals = Sn * (r_surface - r_center) + r_center
  
  return D, D2, Xn, r_vals

#--------------------------------------------------------------

def FCN_Signal(t,tfinal, Amplitude):
    period = tfinal / 2  # Adjust this to control the signal frequency
    signal = np.where(t % period < 0.5 * period, Amplitude, -1 * Amplitude)

    return signal

Instance=Time_Current_Boundary(capacity=capacity, signal=current, initial_surface_flux=surface_flux, Td=Td, Z=Z, tfinal=tfinal, current=current, dt=dt, t=t)  # Here we fill the __init__
boundary = Instance.update_Boundary(current=current,Td=Td,capacity=capacity)

def dZ_dt(Z, t, D, Np):         

    current= FCN_Signal(t,tfinal, Amplitude)  
    derZ_x= D @ Z
    derZ_x[0] = 0
    derZ_x[Np]= (Td/capacity) * current
    d2Z= 1/Td * D @ derZ_x    
    return d2Z

D, D2, Xn, r_vals = Xn_D_D2(r_center=0, r_surface=1, P = Np)

#yy = odeint(dZ_dt, Z0, t, args=(D, Np, FCN_Signal, ))
#yy = odeint(dZ_dt, Z0, t, args=(D, Np, ))
yy = odeint(dZ_dt, Z0, t, args=(D, Np,), rtol=rtol, atol=atol)


# for i in range(len(yy)):
#     print("t:", t[i], "==> yy[", i, "]:", yy[i][0])

C= FCN_Signal(t,tfinal, Amplitude)

#Plot """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(t, C, label='Current')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude [A]')
ax1.grid(True)
ax1.set_title('Current vs Time')
ax1.legend()

x= t
label='SoC_Collocation points'

# Set appropriate x-axis limits to show entire simulation from t=0
ax1.set_xlim(0, t[-1])  # Use the last time point for max limit

for i in range(0, Np+1):
    y = np.array(yy)[:,i][:len(yy)]  # Extract the i-th column
    label = f'SoC_History (Point {i})'  # Create a label for each column
    ax2.plot(x, y, label=label)

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('SoC')
ax2.grid(True)
ax2.set_title('State of Charge at colloc. Points vs Time')
ax2.legend()

# Set appropriate x-axis limits to show entire simulation from t=0
ax2.set_xlim(0, t[-1])  # Use the last time point for max limit

# Adjust layout to prevent overlapping titles
plt.tight_layout()
plt.show()