Please download both programs to be able to run the Chebyshev Collocation method to solve 
Diffusion partial differential equation.
-------------------------------------------------------------------------------------   
The present Python code calculates the effect of the diffusion on the state of charge (Z) in a li-ion 
   battery in a sphere as a particle where Z is a function of distance and time Z= f(x,t).
   The governing equation, a Partial Differential Equation (PDE) has to be solved which is written as:
   
   PDE:
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
