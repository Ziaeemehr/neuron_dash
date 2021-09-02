import numpy as np
import pandas as pd
import pylab as plt
from numpy import pi
from os.path import join
from scipy.integrate import odeint

# import os
# from time import time
# import pylab as plt


def filter_dataframe(df, label):
    """
    filter dataframe

    Parameters
    -------------

    df : Dataframe
        input pandas dataframe
    label :str
        parameter name to be filtered

    return : float
        value of the filtered parameter

    """

    value = df.loc[df['parameter'] == label]["value"].values
    if len(value > 0):
        return value[0]
    else:
        return None


class Montbrio:
    def __init__(self, par, par_current, value) -> bool:

        self.J = filter_dataframe(par, "J")
        r0 = filter_dataframe(par, "r0")
        v0 = filter_dataframe(par, "v0")
        self.eta = filter_dataframe(par, "eta")
        self.tau = filter_dataframe(par, "tau")
        self.Delta = filter_dataframe(par, "Delta")
        self.t_simulation = filter_dataframe(par, "simulation time")
        self.dt = filter_dataframe(par, "dt")
        self.initial_state = [r0, v0]

        self.t_end = filter_dataframe(par_current, "end time")
        self.t_start = filter_dataframe(par_current, "start time")
        self.value = value

        self.amplitude = None
        self.amplitude_end = None
        self.amplitude_start = None
        self.frequency = None
        self.direct_current = None
        self.phase_offset = None

        if value == 1:
            self.amplitude = filter_dataframe(par_current, "amplitude")
        elif value == 2:
            self.amplitude_end = filter_dataframe(par_current, "amplitude end")
            self.amplitude_start = filter_dataframe(
                par_current, "amplitude start")
        else:
            self.frequency = filter_dataframe(par_current, "frequency")
            self.direct_current = filter_dataframe(
                par_current, "direct current")
            self.phase_offset = filter_dataframe(par_current, "phase offset")
            self.amplitude = filter_dataframe(par_current, "amplitude")

    def Iapp(self, t):

        if self.value == 1:  # step
            if (t <= self.t_start) or (t >= self.t_end):
                return 0.0
            else:
                return self.amplitude

        elif self.value == 2:  # ramp
            if (t <= self.t_start) or (t >= self.t_end):
                return 0.0
            else:
                slope = (self.amplitude_end - self.amplitude_start) / \
                    float((self.t_end - self.t_start))
                ramp = self.amplitude_start + (t-self.t_start) * slope
                return ramp
        else:  # sin
            if (t <= self.t_start) or (t >= self.t_end):
                return 0.0
            else:
                phi = t * self.frequency / 1000.0
                phi = phi * 2. * pi + self.phase_offset
                c = np.sin(phi)
                c = (self.direct_current + c * self.amplitude)
                return c

    def drdt(self, r, v):
        return self.Delta / (self.tau * pi * r) + 2 * r * v / self.tau

    def dvdt(self, r, v, t):
        return 1.0/self.tau * (v**2 + self.eta + self.Iapp(t) + self.J *
                               self.tau * r - (pi * self.tau * r)**2)

    def rhs(self, x0, t):
        r, v = x0
        return np.array([self.drdt(r, v), self.dvdt(r, v, t)])

    def simulate(self):

        times = np.arange(0, self.t_simulation, self.dt)
        x = odeint(self.rhs, self.initial_state, times)
        I = [self.Iapp(t) for t in times]

        return {"t": times, "r": x[:, 0], "v": x[:, 1], "I": I}

    def vector_filed(self,
                     x=[0, 4],
                     y=[0, 4],
                     nx=50,
                     ny=50,
                     I0=3.0):

        r0 = np.linspace(x[0], x[1], nx)
        v0 = np.linspace(y[0], y[1], ny)
        r, v = np.meshgrid(r0, v0)

        drdt = self.drdt(r, v)
        dvdt = 1.0/self.tau * (v**2 + self.eta + I0 + self.J *
                               self.tau * r - (pi * self.tau * r)**2)

        return (r, v, drdt, dvdt)

    def nullclines(self, x=[0, 2], dx=0.001, I=3.0):

        def rnull(r):
            return -self.Delta/(2 * pi * r**2)

        def vnull(r, I):
            v = np.sqrt(-self.eta/self.tau-self.J*r-I /
                        self.tau + pi**2 * r**2*self.tau)
            return -v, v

        r = np.arange(x[0], x[1], dx)
        v0 = rnull(r)
        v1, v2 = vnull(r, I)

        return {"r": r, "rnull": v0,  "vnull1": v1, "vnull2": v2}


if __name__ == "__main__":

    from plotly.figure_factory import create_streamline
    data = pd.read_csv(join("..", "apps", "montbrio.csv"))
    df_par = data.loc[data["category"] == "par"].reset_index()
    df_current = data.loc[data["category"] == "cur"].reset_index()
    value = 1

    MB = Montbrio(df_par, df_current, value)
    data = MB.simulate()
    x = [0.1, 2]
    y = [0.1, 1.5]
    r, v, dr, dv = MB.vector_filed(x=x, y=y, nx=10, ny=10)
    # plt.streamplot(r, v, dr, dv)
    # plt.savefig("0.png")
    # fig = create_streamline(r, v, dr, dv)
    # fig.update_layout(yaxis_range=y,
    #                   xaxis_range=x)
    # fig.update_traces(line_color="black")

    # fig.show()
    # print(data['r'])

    # ax.streamplot(phi1, phi2,
    #               dphi1_dt, dphi2_dt,
    #               color='k',
    #               linewidth=0.5,
    #               cmap=plt.cm.autumn)
    # ax.quiver(phi1, phi2, dphi1_dt, dphi2_dt)


# def simulate_montbrio(par, par_current, value):

#     def Iapp(t):

#         # dur = t_end - t_start

#         if value == 1:  # step
#             if (t <= t_start) or (t >= t_end):
#                 return 0.0
#             else:
#                 return amplitude

#         elif value == 2:  # ramp
#             if (t <= t_start) or (t >= t_end):
#                 return 0.0
#             else:
#                 slope = (amplitude_end - amplitude_start) / \
#                     float((t_end - t_start))
#                 ramp = amplitude_start + (t-t_start) * slope
#                 return ramp
#         else:
#             phi = t * frequency/1000
#             phi = phi * 2. * pi + phase_offset
#             c = np.sin(phi)
#             c = (direct_current + c * amplitude)
#             return c

#     def rhs(x0, t):
#         r, v = x0
#         drdt = Delta / (tau * pi * r) + 2 * r*v / tau
#         dvdt = 1.0/tau * (v**2 + eta + Iapp(t) + J *
#                           tau * r - (pi * tau * r)**2)

#         return np.array([drdt, dvdt])

#     J = filter_dataframe(par, "J")
#     r0 = filter_dataframe(par, "r0")
#     v0 = filter_dataframe(par, "v0")
#     eta = filter_dataframe(par, "eta")
#     tau = filter_dataframe(par, "tau")
#     Delta = filter_dataframe(par, "Delta")
#     t_simulation = filter_dataframe(par, "simulation time")
#     dt = filter_dataframe(par, "dt")
#     initial_state = [r0, v0]

#     t_end = filter_dataframe(par_current, "end time")
#     t_start = filter_dataframe(par_current, "start time")

#     if value == 1:
#         amplitude = filter_dataframe(par_current, "amplitude")
#     elif value == 2:
#         amplitude_end = filter_dataframe(par_current, "amplitude end")
#         amplitude_start = filter_dataframe(par_current, "amplitude start")
#     else:
#         frequency = filter_dataframe(par_current, "frequency")
#         direct_current = filter_dataframe(par_current, "direct current")
#         phase_offset = filter_dataframe(par_current, "phase offset")
#         amplitude = filter_dataframe(par_current, "amplitude")

#     times = np.arange(0, t_simulation, dt)
#     x = odeint(rhs, initial_state, times)
#     I = [Iapp(t) for t in times]

#     return {"t": times, "r": x[:, 0], "v": x[:, 1], "I": I}
