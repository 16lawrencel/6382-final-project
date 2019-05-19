# Symbolically calculate derivatives to use for DMOC
from sympy import *

mh = Symbol('mh')
m = Symbol('m')
l = Symbol('l')
a = Symbol('a')
b = Symbol('b')
g = Symbol('g')
h = Symbol('h')

def simplediff(expr, var):
    return simplify(diff(expr, var))

def L(th_st, dth_st, th_sw, dth_sw):
    # Lagrangian of compass gait system
    return 0.5*(mh*l*l + m*a*a + m*l*l) * dth_st**2 + 0.5*m*b*b * dth_sw**2 \
            - m * l * b * dth_st * dth_sw * cos(th_sw - th_st) \
            - g * (m*a + m*l + mh*l) * cos(th_st) \
            + m*g*b * cos(th_sw)

def Ld(th_st_0, th_sw_0, th_st_1, th_sw_1):
    return h*L((th_st_0 + th_st_1) / 2, (th_st_1 - th_st_0) / h, (th_sw_0 + th_sw_1) / 2, (th_sw_1 - th_sw_0) / h)

def D2_L(th_st, dth_st, th_sw, dth_sw):
    L_expr = L(th_st, dth_st, th_sw, dth_sw)
    return [simplediff(L_expr, dth_st), simplediff(L_expr, dth_sw)]

def D1_Ld(th_st_0, th_sw_0, th_st_1, th_sw_1):
    Ld_expr = Ld(th_st_0, th_sw_0, th_st_1, th_sw_1)
    return [simplediff(Ld_expr, th_st_0), simplediff(Ld_expr, th_sw_0)]

def D2_Ld(th_st_0, th_sw_0, th_st_1, th_sw_1):
    Ld_expr = Ld(th_st_0, th_sw_0, th_st_1, th_sw_1)
    return [simplediff(Ld_expr, th_st_1), simplediff(Ld_expr, th_sw_1)]

th_st = Symbol('th_st')
dth_st = Symbol('dth_st')
th_sw = Symbol('th_sw')
dth_sw = Symbol('dth_sw')
print "D2_L:", D2_L(th_st, dth_st, th_sw, dth_sw)

th_st_0 = Symbol('th_st_0')
th_sw_0 = Symbol('th_sw_0')
th_st_1 = Symbol('th_st_1')
th_sw_1 = Symbol('th_sw_1')

print "D1_Ld:", D1_Ld(th_st_0, th_sw_0, th_st_1, th_sw_1)
print "D2_Ld:", D2_Ld(th_st_0, th_sw_0, th_st_1, th_sw_1)
