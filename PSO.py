# Traigo las librerías
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd 
import operator

# Defino la función de costo que quiero optimizar (Vamos a maximizarla)
def ML_GPD(y,gamma,sigma):
  y = np.array(y)
  if gamma < -1 or gamma*max(y)/sigma <=-1:
    print(" ===== Los parámetros de la función no son válidos =====")
    print("gamma = {} ----- sigma = {}".format(gamma,sigma))
    return None
  elif gamma ==0:
    p1 = len(y)*np.log(sigma)
    p2 = (1/sigma)*np.sum(y)
  else:
    p1 = len(y)*np.log(sigma)
    p2 = (1+1/gamma)*np.sum(np.log(1+gamma*y/sigma))
    return -1*p1 - p2
  

# Defino la clase que va a ejecutar gbest PSO

class GBestPSO():
  def __init__(self,y,n_particulas = 100):

    # Guardo y
    self.y = y

    # Creo las partículas. Tienen que estar dentro de A = {gamma <= -1 ; max(y) > -sigma/gamma} .
    # Posiblemente lo mejor sea distribuir las partículas uniformemente entre -1 <= gamma < 3 y 0 < sigma <= 4.
    # Las combinaciones no válidas de gamma y sigma se eliminan.
    gammas = np.arange(-1,3,4/np.sqrt(n_particulas))
    sigmas = -1 * np.arange(-4,0,4/np.sqrt(n_particulas))
    self.posiciones = [[i,j] for i in gammas for j in sigmas if i*max(y)/j >-1]

    # Vector velocidades
    self.velocidades = [[np.random.uniform(-0.5,0.5,1),np.random.uniform(-0.5,0.5,1)] for i in self.posiciones]

    # Creo los valores aleatorios del PSO
    self.r1 = np.random.uniform(0,1,1)
    self.r2 = np.random.uniform(0,1,1)

    # Creo las listas que van a guardar los mínimos personales y el mínimo global
    self.max_per = []
    self.max_glob = []



  def fit (self,c1 = 0.1,c2 = 0.2,iter = 100000):
    
    # El primer paso es ejecutar ver el costo en las funciones y guardarlo en el mínimo personal
    self.max_per = [[ML_GPD(self.y,i[0],i[1]),i[0],i[1]] for i in self.posiciones] 
    valores = [i[0] for i in self.max_per]
    # Guardo el máximo global
    maximo = max(valores)
    for i in self.max_per:
      if i[0] == maximo:
        self.max_glob = i
    

    # Inicializo un contador de iter
    contador = 0

    # Hago el loop
    while iter-contador>0:
      if contador%200 ==0:
        print("Maximo Global")
        print(self.max_glob)
        Promedio_Velocidades = 0
        for i in range(len(self.velocidades)):
          Promedio_Velocidades = Promedio_Velocidades + (self.velocidades[i][0]**2 + self.velocidades[i][1]**2)**(0.5)
        
        print("Promedio Velocidades")
        print(Promedio_Velocidades/len(Promedio_Velocidades))
        print("")
      # Actualizo contador
      contador +=1

      # Actualizo posiciones

      for i in range(len(self.posiciones)):
        if (self.posiciones[i][0] + self.velocidades[i][0])*(max(self.y))/(self.posiciones[i][1] + self.velocidades[i][1]) >-1 and self.posiciones[i][0] + self.velocidades[i][0]>=-1:
          
          self.posiciones[i] = [self.posiciones[i][0] + self.velocidades[i][0] , self.posiciones[i][1] + self.velocidades[i][1]]

      # Actualizo máximos personales
      for i in range(len(self.posiciones)):
        if ML_GPD(self.y,self.posiciones[i][0],self.posiciones[i][1])>self.max_per[i][0]:
          self.max_per[i] = [ML_GPD(self.y,self.posiciones[i][0],self.posiciones[i][1])[0],self.posiciones[i][0][0],self.posiciones[i][1][0]]

      # Actualizo máximo global 
      valores = [i[0] for i in self.max_per]
      # Guardo el máximo global
      maximo = max(valores)
      for i in self.max_per:
        if i[0] == maximo:
          self.max_glob = i
      
      # Actualizo velocidades
      for i in range(len(self.max_per)):
        #print(self.max_per)
        #print(self.max_per[i])
        #print(self.posiciones[i][0])
        #print(self.max_glob[1])
        self.velocidades[i][0] = self.velocidades[i][0] + c1 * self.r1* ( self.max_per[i][1] - self.posiciones[i][0] ) + c2 * self.r2 * ( self.max_glob[1] - self.posiciones[i][0] )
        self.velocidades[i][1] = self.velocidades[i][1] + c1 * self.r1* ( self.max_per[i][2] - self.posiciones[i][1] ) + c2 * self.r2 * ( self.max_glob[2] - self.posiciones[i][1] )



# Defino la clase que va a ejecutar lbest PSO

class LBestPSO():
  def __init__(self,y,n_particulas = 100,n_grupo=3):

    # Guardo y
    self.y = y

    # Creo las partículas. Tienen que estar dentro de A = {gamma <= -1 ; max(y) > -sigma/gamma} .
    # Posiblemente lo mejor sea distribuir las partículas uniformemente entre -1 <= gamma < 3 y 0 < sigma <= 4.
    # Las combinaciones no válidas de gamma y sigma se eliminan.
    gammas = np.arange(-1,3 + 4/np.sqrt(n_particulas),4/np.sqrt(n_particulas))
    sigmas = -1 * np.arange(-4,0,4/np.sqrt(n_particulas))
    self.posiciones = [[i,j] for i in gammas for j in sigmas if i*max(y)/j >-1]

    # Vector velocidades
    self.velocidades = [[np.random.uniform(-0.5,0.5,1),np.random.uniform(-0.5,0.5,1)] for i in self.posiciones]

    # Creo los valores aleatorios del PSO
    self.r1 = np.random.uniform(0,1,1)
    self.r2 = np.random.uniform(0,1,1)

    # Creo las listas que van a guardar los máximos personales y los máximos locales para cada partícula
    self.max_per = []
    self.max_loc = []

    # Guardo las variables para la cantidad de particulas por grupo local

    self.n_grupo = n_grupo



  def fit (self,c1 = 0.1,c2 = 0.2,iter = 250):
    
    # El primer paso es ejecutar ver el costo en las funciones y guardarlo en el máximo personal
    self.max_per = [[ML_GPD(self.y,i[0],i[1]),i[0],i[1]] for i in self.posiciones] 

    # Guardo los máximos locales
    distancias = []
    for i in range(len(self.posiciones)):
      Aux = []
      for j in range(len(self.posiciones)):
        if i != j:
          Aux.append([((self.posiciones[i][0] - self.posiciones[j][0])**2 + (self.posiciones[i][1] - self.posiciones[j][1])**2)**(0.5),j])
      distancias.append(sorted(Aux, key=operator.itemgetter(0)))


    distancias = [i[0:self.n_grupo] for i in distancias]
    Indices_Cercanos = []
    for i in distancias:
      Aux = [i[j][1] for j in range(self.n_grupo)]
      Indices_Cercanos.append(Aux)
    for i in range(len(self.max_per)):
      valor = -10000000
      gamma = 0
      sigma = 0
      for j in Indices_Cercanos[i]:
        if valor<self.max_per[j][0]:
          valor = self.max_per[j][0]
          gamma = self.max_per[j][1]
          sigma = self.max_per[j][2]

      self.max_loc.append([valor,gamma,sigma])

    


    valores = [i[0] for i in self.max_loc]
    maximo = max(valores)
    for i in self.max_per:
      if i[0] == maximo:
        self.max_glob = i
    

    # Inicializo un contador de iter
    contador = 0

    # Hago el loop
    while iter-contador>0:
        

      # Actualizo contador
      contador +=1

      # Actualizo posiciones

      for i in range(len(self.posiciones)):
        if (self.posiciones[i][0] + self.velocidades[i][0])*(max(self.y))/(self.posiciones[i][1] + self.velocidades[i][1]) >-1 and self.posiciones[i][0] + self.velocidades[i][0]>=-1:
          self.posiciones[i] = [self.posiciones[i][0] + self.velocidades[i][0] , self.posiciones[i][1] + self.velocidades[i][1]]

      # Actualizo máximos personales
      for i in range(len(self.posiciones)):
        if ML_GPD(self.y,self.posiciones[i][0],self.posiciones[i][1])>self.max_per[i][0]:
          self.max_per[i] = [ML_GPD(self.y,self.posiciones[i][0],self.posiciones[i][1])[0],self.posiciones[i][0][0],self.posiciones[i][1][0]]

      # Actualizo máximos locales 
      # Guardo los máximos locales
      distancias = []
      for i in range(len(self.posiciones)):
        Aux = []
        for j in range(len(self.posiciones)):
          if i != j:
            Aux.append([((self.posiciones[i][0] - self.posiciones[j][0])**2 + (self.posiciones[i][1] - self.posiciones[j][1])**2)**(0.5),j])
        distancias.append(sorted(Aux, key=operator.itemgetter(0)))


      distancias = [i[0:3] for i in distancias]
      Indices_Cercanos = []
      self.max_loc = []
      for i in distancias:
        Aux = [i[0][1],i[1][1],i[2][1]]
        Indices_Cercanos.append(Aux)
      for i in range(len(self.max_per)):
        valor = -10000000
        gamma = 0
        sigma = 0
        for j in Indices_Cercanos[i]:
          if valor<self.max_per[j][0]:
            valor = self.max_per[j][0]
            gamma = self.max_per[j][1]
            sigma = self.max_per[j][2]

        self.max_loc.append([valor,gamma,sigma])

        
      # Guardo el máximo global
      valores = [i[0] for i in self.max_loc]
      maximo = max(valores)
      for i in self.max_per:
        if i[0] == maximo:
          self.max_glob = i
      
      # Actualizo velocidades
      for i in range(len(self.max_per)):
        #print(self.max_per)
        #print(self.max_per[i])
        #print(self.posiciones[i][0])
        #print(self.max_glob[1])
        self.velocidades[i][0] = self.velocidades[i][0] + c1 * self.r1* ( self.max_per[i][1] - self.posiciones[i][0] ) + c2 * self.r2 * ( self.max_loc[i][1] - self.posiciones[i][0] )
        self.velocidades[i][1] = self.velocidades[i][1] + c1 * self.r1* ( self.max_per[i][2] - self.posiciones[i][1] ) + c2 * self.r2 * ( self.max_loc[i][2] - self.posiciones[i][1] )
    #print("Maximo Global")
    #print(self.max_glob)
    return self.max_glob
