import numpy as np
class Time_Current_Boundary:
    """
    This class calculates the provide Current profile, time, and boundary conditions.
    """

    def __init__(self, signal=None, capacity=None, initial_surface_flux=None, Td=None, Z=None, tfinal=None, current=None, dt=None,t=None, Amplitude=None):
        """
        Initializes the Coulomb class with battery capacity and initial surface_flux.

        Args:
            capacity: Battery capacity in units consistent with current.
            initial_surface_flux: Initial state of charge (0 to 1).
            Td: Diffusion time constant.
        """
        self.capacity = capacity
        self.surface_flux = initial_surface_flux
        self.time = 0  # Track total elapsed time
        self.time_history = [self.time]  # Initialize with initial time (t=0)
        self.surface_flux_history = [self.surface_flux]  # Initialize with initial surface_flux
        self.Td = Td
        self.Z = Z
        self.tfinal = tfinal
        self.current = current
        self.dt = dt
        self.t = t
        self.signal=signal
        self.Amplitude=Amplitude
        
    def FCN_Signal(self, Amplitude):
        period = self.tfinal / 4  # Adjust this to control the signal frequency
        self.signal = np.where(self.t % period < 0.5 * period, Amplitude, -1 * Amplitude)        
        return self.signal

    def update_Boundary(self,current,Td,capacity):
        self.surface_flux = current * Td / capacity
        self.time += self.dt  # Update time after calculating surface_flux
        self.time_history.append(self.time)
        self.surface_flux_history.append(self.surface_flux)
        #print(self.surface_flux)
        return self.surface_flux