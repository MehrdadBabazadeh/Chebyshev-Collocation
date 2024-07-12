"""
Please download both Python programs to be able to run the Chebyshev Collocation method to solve 
Diffusion partial differential equation.
-------------------------------------------------------------------------------------   
The present Python code calculates the effect of the diffusion on the state of charge (Z) in a li-ion 
   battery in a sphere as a particle where Z is a function of distance and time Z= f(x,t).
   The governing equation, a Partial Differential Equation (PDE) has to be solved which is written as:
   
   PDE: includes derivatives concerning two different variables, a spatial variable and a time variable
      Td*dZ/dt = d2Z/dx2  where d2Z/dx2 is the second partial derivative in terms of the distance x from center
   Boundary conditions:   
      dZ/dx at center = 0
      dZ/dx at surface = surface flux
   
   An approximate method of discretization, the Chebyshev Collocation Method is used to simplify the solution. 
   The program starts after the number of collocation points (N_collocation) is set (default value is 6). 
   
   Based on an inventory arrangement of the problem and matrix operations, the Backward-Euler method 
   can be applied to calculate vector Z(t).
   
   Current profile, time, and boundary conditions are updated in a Class named: Time_Current_Boundary
   
   Please note: 
   1. The first Derivative matrix (D) is multiplied by 2 in the code. Therefore, there is no need
      to apply number 4 at the discrete 
   form in the square of D (D2). Therefore, the discrete equation will be:
   Td* dZ/dt= D2*Z 
   where Td is the known diffusion time constant.
   2. Other methods based on ODEINT in Python are not stable while calculating with different time steps
      or the number of collocation points
   3. Current limitation has been applied to prevent Z from exceeding the permitted area (0<Z<1)
   Copyright (C) Dr. Mehrdad Babazadeh -  WMG, University of Warwick, UK. 11-06-2024
   All Rights Reserved     
"""

import numpy as np
import matplotlib.pyplot as plt
from MB_Time_Current_Boundary import Time_Current_Boundary

import os
os.system('cls' if os.name == 'nt' else 'clear')   # To clear terminal

#Settings--------------------------------------------------------------
N_collocation = 6
Np = N_collocation - 1 

dt= 1                # time step
tfinal= 3600 *6      # 360*5 ; \With small values batery may not be chsarged

Td = 580             # 580 ; Diffusion time constant
capacity= 4.9 * 3600   # 4.9 [A.h] ==> [4.9* 3600 A.sec]

#SoC initial--------------------------------------------------------------
Init = 0.5  # Initial condition of SoC only at collocation points, not in the center and surface 
Z1= np.ones(Np-1) * Init
Z = np.ones(Np+1) * Init

SoC_Surface= Z[Np]
surface_flux = 0
Z0=Z.copy()
SoC_History=[]
Zbar_History=[]
ZSurface_History=[]
ZDiff_History=[]
current= 0
Amplitude =10
dZ0=np.zeros(Np+1)

#time definition--------------------------------------------------------------
if dt>=1:
  t = np.linspace(0, tfinal, tfinal) 
else:
  t = np.arange(0, tfinal + dt, dt)
#--------------------------------------------------------------

def Xn_D_D2(r_center=0, r_surface=1, P = Np):

  Sn=np.zeros(P+1)
  Xn=np.zeros(P+1)
  D = np.zeros((P+1, P+1))
  Dx=np.zeros((P+1, P+1))

  Xn[0]=1  
  for j in range (0,P+1):
    Sn[j] = -(np.cos(j*np.pi/Np))
    Xn[j] = 1-0.5*(1+ np.cos(j*np.pi/Np))   # I have subtracted from 1 to show points as starting from 0 to 1

  for i in range (0,P+1):
    for j in range (0,P+1):  
      if (i==j and i==0):
         D[i][j]= -(2*P**2+1)/6
      elif (i==j and i==P):
         D[i][j]= (2*P**2+1)/6
      elif (i==j and i<=P-1):
         D[i][j]=round (-Sn[i]/(2*(1-Sn[i]**2)),4)
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

  D = D*2   #Derivative Matrix
  #print("D:", D)
  # Transform collocation points to actual radius
  r_vals = Sn * (r_surface - r_center) + r_center

  Dx[1:Np][:] = D[1:Np][:]
  #print("Dx:", Dx)
  Dm = D @ Dx
  
  A1 = np.eye(Np+1)- Dm*dt/Td
  A1 = np.linalg.inv(A1)
  
  # Transform collocation points to actual radius
  r_vals = Xn * (r_surface - r_center) + r_center
  
  print("________________________________________________________________________________________")    
  print("Number of Collocation points:", Np+1)
  print("Xn (Collocation points):", Xn)
  print("________________________________________________________________________________________")    

  return D, Dx, Dm, Xn, r_vals, A1

#--------------------------------------------------------------
D, Dx, Dm, Xn, r_vals, A1 = Xn_D_D2(r_center=0, r_surface=1, P = Np)

Instance=Time_Current_Boundary(capacity=capacity, signal=current, initial_surface_flux=surface_flux, Td=Td, Z=Z, tfinal=tfinal, current=current, dt=dt, t=t)  # Here we fill the __init__
current_profile = Instance.FCN_Signal(Amplitude)
   
i=0
flag_High=0
flag_Low=0

for current in current_profile:   
  SoC_History.append(Z.copy())  # Append the current Z vector to SoC_History
  Zbar=np.mean(Z)
  Zbar_History.append(Zbar)
  ZSurface=Z[Np]
  ZSurface_History.append(ZSurface)
  ZDiff=ZSurface-Zbar
  ZDiff_History.append(ZDiff)
  
  #print("ZSurface:",ZSurface)

  #print("Zbar:",Zbar)
  if current<0:
     flag_High=0
  if current>0:
     flag_Low=0
  
  if (np.any(Z>=1) and current>0) or flag_High==1:
     current= 0 #-Amplitude
     current_profile[i]=0
     flag_High=1
     flag_Low=0
     
  if (np.any(Z<=0) and current<0) or flag_Low==1 :
     current= 0 #Amplitude     
     current_profile[i]=0
     flag_High=0
     flag_Low=1
          
  i=i+1
  
  surface_flux = Instance.update_Boundary(current=current,Td=Td,capacity=capacity)

  t1= Instance.time
  
  dZ0[0] = 0               # initial value
  dZ0[Np]= surface_flux   # initial value
  
  A= D @ dZ0
  Z= A1 @ (A*dt/Td+ Z)

#Plot """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
# Access subplots using indexing
ax1 = axes[0, 0]  # Top-left subplot
ax2 = axes[0, 1]  # Top-right subplot
ax3 = axes[1, 0]  # Bottom-left subplot
ax4 = axes[1, 1]  # Bottom-right subplot


ax1.plot(t, current_profile, label='Current')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude [A]')
ax1.grid(True)
ax1.set_title('Controlled Current vs. Time')
ax1.legend(fontsize=6,loc='center right')

# Set appropriate x-axis limits to show the entire simulation from t=0
ax1.set_xlim(0, t[-1])  # Use the last time point for max limit

for i in range(0, Np+1):
    y = np.array(SoC_History)[:,i][:len(SoC_History)]  # Extract the i-th column
    ax2.plot(t, y, label= f'SoC at point {i}')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('SoC')
ax2.grid(True)
ax2.set_title('SoC at Collocation points between center and surface vs. Time')
ax2.legend(fontsize=6,loc='center right')
ax2.set_xlim(0, t[-1])  # Use the last time point for max limit


ax3.plot(t, ZSurface_History, label='SoC Surface')
ax3.plot(t, Zbar_History, label='SoC average')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('SoC average')
ax3.grid(True)
ax3.set_title('SoC(Surface) and SoC(average) vs. Time')
ax3.legend(fontsize=6,loc='center right')
ax3.set_xlim(0, t[-1])  # Use the last time point for max limit



ax4.plot(t, ZDiff_History, label='SoC Diffusion')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('SoC Diffusion')
ax4.grid(True)
ax4.set_title('SoC Diffusion vs. Time')
ax4.legend(fontsize=6,loc='center right')
# Set appropriate x-axis limits to show the entire simulation from t=0
ax4.set_xlim(0, t[-1])  # Use the last time point for max limit

# Adjust layout to prevent overlapping titles
plt.tight_layout()
plt.show()