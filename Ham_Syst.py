import math
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
# ВХОДНЫЕ ДАННЫЕ
p0 = float(input('введите начальный импульс точки p0 = '))
x0 = float(input('введите начальную координату точки x0 = '))
t0 = 0 # неизменяемый параметр для данного вида функции f(t)
tn = 2000 # неизменяемый параметр для данного вида функции f(t)
n = 7500000 # можно сделать ввод пользователя, но относительно точное решение требует точности порядка 10**(-3) - 10**(-2)
h = (tn-t0)/n
tk = np.arange(t0, tn + h, h) # разбиение временного отрезка
# для графика f(t)
c = 1
b = 0.1
a = 2
w = 0.008
t1 = 500
# Рунге-Кутт
def fr(t) :
  return (1/(1+math.exp(-2*w*(t-t1))))+(1/(1+math.exp(2*w*(t-3*t1))))-1
def f1(p, x, t): # f1(p,x,t) -> [dx(t)/dt=f1(p,x,t)]
  return p
def f2(p, x, t): # f2(p,x,t) -> [dp(t)/dt=f2(p,x,t)]
  return c*fr(t)-b*x-a*math.sin(x)
xk = [x0]
pk = [p0]
for i in tk:
  k1 = f2(p0, x0, i)*h
  m1 = f1(p0, x0, i)*h
  k2 = f2(p0+m1/2, x0+k1/2, i+h/2)*h
  m2 = f1(p0+m1/2, x0+k1/2, i+h/2)*h
  k3 = f2(p0+m2/2, x0+k2/2, i+h/2)*h
  m3 = f1(p0+m2/2, x0+k2/2, i+h/2)*h
  k4 = f2(p0+m3, x0+k3, i+h/2)*h
  m4 = f1(p0+m3, x0+k3, i+h/2)*h
  q1 = (k1+2*k2+2*k3+k4)/6
  q2 = (m1+2*m2+2*m3+m4)/6
  p0 = p0+q1
  x0 = x0+q2
  pk.append(p0)
  xk.append(x0)
plt.figure(1) # ПОСТРОЕНИЕ РЕШЕНИЯ
plt.plot(xk, pk)
plt.xlabel('x')
plt.ylabel('p (x)')
from scipy import integrate # ПРОВЕРКА РЕШЕНИЯ
def f(y, t): # РЕШЕНИЕ ИЩЕТСЯ В ВИДЕ СТОБЦА y=[x,p],правая часть уравнения имеет вид A(y,t) = [y11,y22], т.о. решается ДУ: dy/dt=A(y,t)
  x = y[0]
  p = y[1]
  y11 = p
  y22 = c*fr(t)-b*x-a*math.sin(x)
  return [y11, y22]
sol = integrate.odeint(f, [x0, p0], tk) # ИНТЕГРИРУЕМ СИСТЕМУ ДУ
X_sol = []
P_sol = []
for i in sol:
  X_sol.append(i[0])
  P_sol.append(i[1])
plt.figure(2) # Р(X) ИЗ ПРОВЕРКИ
plt.plot(X_sol, P_sol)
plt.xlabel('x')
plt.ylabel('p (x)')
fig = plt.figure(3) # ДВИЖЕНИЕ ТОЧКИ
camera = Camera(fig)
plt.xlabel('x')
plt.ylabel('p (x)')
for i in range(0, n+250000, 250000):
    plt.scatter(xk[i], pk[i], s=50, color='red')
    plt.plot(xk, pk, color='cornflowerblue')
    camera.snap()
animation = camera.animate()
plt.show()