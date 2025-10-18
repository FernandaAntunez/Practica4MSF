"""
Práctica: Sistema Cardiovascular

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Jovita Maria Fernanda Antunez Rubio
Número de control: 22211745
Correo institucional:L22211745@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""



# Instalar librerias en consola
#!pip install control
#!pip install slycot


# Librerías para cálculo numérico y generación de gráficas
import numpy as np
import math as m 
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

u= np.array(pd.read_excel('signal.xlsx', header=None))
# Datos de la simulación
x0, t0, tend, dt, w, h= 0,0,10,47E-3, 7, 3.5
N= round((tend - t0)/dt)+1
t = np.linspace(t0,tend,N)
u = np.reshape(signal.resample(u,len(t)),-1)

def cardio (Z,C,R,L):
    num =[L*R,R*Z]
    den =[C*L*R*Z,L*R+L*Z, R*Z]
    sys = ctrl.tf(num,den)
    return sys

#Función  de tranferencia: Normotenso
Z,C,R,L= 0.033, 1.5, 0.95, 0.01
sysnormo= cardio(Z,C,R,L)
print(f'Función de transferencia del Normotenso: {sysnormo}')
   
#Función  de tranferencia: Hipotenso
Z,C,R,L= 0.02, 0.25, 0.6, 0.05
syshipo= cardio(Z,C,R,L)
print(f'Función de transferencia del Hipotenso: {syshipo}') 

#Función  de tranferencia: Hipertenso
Z,C,R,L= 0.04, 2.5, 1.4, 0.02
syshiper= cardio(Z,C,R,L)
print(f'Función de transferencia del Hipertenso: {syshiper}') 
    


#Respuestas en lazo abierto
_,Pp0 = ctrl.forced_response(sysnormo,t,u,x0)
_,Pp1 = ctrl.forced_response(syshipo,t,u,x0)
_,Pp2 = ctrl.forced_response(syshiper,t,u,x0)

clr1= np.array([100,13,95])/255
clr2= np.array ([217, 22, 86])/255
clr6= np.array ([255,178, 0])/255

fg1 = plt.figure()
plt.plot(t,Pp0,'-',linewidth=1,color=clr1,label='Pp(t):Normotenso')
plt.plot(t,Pp1,'-',linewidth=1,color=clr2,label='Pp(t):Hipotenso')
plt.plot(t,Pp2,'-',linewidth=1,color=clr6,label='Pp(t):Hipertenso')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('Pp(t) [V]')
plt.ylabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,0.2),loc='center',ncol=3)
plt.show()
fg1.set_size_inches(w,h)
fg1.tight_layout()
fg1.savefig('sistema cardiovascular phyton.png',dpi=600, bbox_inches='tight')
fg1.savefig('sistema cardiovascular phyton.pdf')

def controlador (kP,kI,sys):
    Cr = 1e-6
    Re = 1/(kI*Cr)
    Rr = kP*Re
    numPI = [Rr*Cr,1]
    denPI = [Re*Cr,0]
    PI = ctrl.tf(numPI, denPI)
    X = ctrl.series(PI,sys)
    sysPI = ctrl.feedback(X,1,sign=-1)
    return sysPI

hipoPI = controlador(3.6723004806354e-05,3751.80641906569,syshipo)
hiperPI = controlador(10,3751.80641906569,syshiper)

#Respuestas en lazo cerrado
_,Pp3 = ctrl.forced_response(hipoPI,t,Pp0,x0)
_,Pp4 = ctrl.forced_response(hiperPI,t,Pp0,x0)

    
fg2 = plt.figure()
plt.plot(t,Pp0,'-',linewidth=1,color=clr1,label='Pp(t):Normotenso')
plt.plot(t,Pp1,'-',linewidth=1,color=clr2,label='Pp(t):Hipotenso')
plt.plot(t,Pp3,'--',linewidth=1,color=clr6,label='Pp(t):Hipotenso PI')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('Pp(t) [V]')
plt.ylabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,0.2),loc='center',ncol=3)
plt.show()
fg2.set_size_inches(w,h)
fg2.tight_layout()
fg2.savefig('sistema cardiovascularHipoPI phyton.png',dpi=600, bbox_inches='tight')
fg2.savefig('sistema cardiovascularHIPOPI phyton.pdf')

fg3 = plt.figure()
plt.plot(t,Pp0,'-',linewidth=1,color=clr1,label='Pp(t):Normotenso')
plt.plot(t,Pp2,'-',linewidth=1,color=clr2,label='Pp(t):Hipertenso')
plt.plot(t,Pp4,'--',linewidth=1,color=clr6,label='Pp(t):Hipertenso PI')
plt.grid(False)
plt.xlim(0,10); plt.xticks(np.arange(0,11,1))
plt.ylim(-0.6,1.4); plt.yticks(np.arange(-0.6,1.6,0.2))
plt.xlabel('Pp(t) [V]')
plt.ylabel('t [s]')
plt.legend(bbox_to_anchor=(0.5,0.2),loc='center',ncol=3)
plt.show()
fg3.set_size_inches(w,h)
fg3.tight_layout()
fg3.savefig('sistema cardiovascularHiperPI phyton.png',dpi=600, bbox_inches='tight')
fg3.savefig('sistema cardiovascular phytonHiperPI.pdf')