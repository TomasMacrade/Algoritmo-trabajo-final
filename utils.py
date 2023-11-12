import numpy as np
from PSO import LBestPSO

# Funcion de densidad acumulada de GPD
def GPD_Acum(x,k,sigma):
    return 1 - (1-k*x/sigma)**(1/k)

# Calculo de estadístico de Anderson-Darling
def A_2(y,k,sigma):
    n = len(y)
    y.sort()
    suma = 0
    for i in range(n):
        j = i+1
        suma = suma + (2*j-1)*(np.log(GPD_Acum(y[i],k,sigma)) + np.log(1-GPD_Acum(y[n-1-i],k,sigma)))
    return -n-suma*(1/n)

# Calculo de estadístico de Cramer-Von Mises
def W_2(y,k,sigma):
    n = len(y)
    y.sort()
    suma = (1/(12*n))
    for i in range(n):
        j = i+1
        suma = suma + (GPD_Acum(y[i],k,sigma) - (2*j-1)/(2*n))**2
    return suma

# Función que te dice si el test está aprobado con p-value >0.25
# Estos valores se sacan de Choulakian, V., and M.A. Stephens. 2001. 
# Goodness-of-ft tests for the generalized Pareto distribution. 
# Technometrics 43 (4): 478–484.

def Test_P_01(k,A_2,W_2):
    if k>=-1 and k<-0.7 and A_2<=0.471 and W_2<=0.067:
        return "Se aprobó el test!"
    elif k>=-0.7 and k<-0.35 and A_2<=0.499 and W_2<=0.072:
        return "Se aprobó el test!"
    elif k>=-0.35 and k<-0.15 and A_2<=0.534 and W_2<=0.078:
        return "Se aprobó el test!"
    elif k>=-0.15 and k<-0.05 and A_2<=0.550 and W_2<=0.081:
        return "Se aprobó el test!"
    elif k>=-0.05 and k<0.05 and A_2<=0.569 and W_2<=0.086:
        return "Se aprobó el test!"
    elif k>=0.05 and k<0.15 and A_2<=0.591 and W_2<=0.089:
        return "Se aprobó el test!"
    elif k>=0.15 and k<0.25 and A_2<=0.617 and W_2<=0.094:
        return "Se aprobó el test!"
    elif k>=0.25 and k<0.35 and A_2<=0.649 and W_2<=0.1:
        return "Se aprobó el test!"
    elif k>=0.35 and k<0.45 and A_2<=0.688 and W_2<=0.107:
        return "Se aprobó el test!"
    elif k>=0.45  and A_2<=0.735 and W_2<=0.116:
        return "Se aprobó el test!"
    else:
        return "No se aprobó el test!"
    
# Función que recibe un dataframe de pandas con una variable y la fecha en que fue tomada 
# y devuelve una lista con datos cuya separación en dias sea mayor a n_dias.
# En caso de conflicto, se queda con el valor más grande y su fecha.
def Datos_Indep (Datos,n_dias=4):
    Años = []
    for i in Datos.values.tolist():
        Años.append(i[0][:4])
    Años = [int(i) for i in list(set(Años))]
    Años.sort()
    Datos_Independientes = []
    for Año in Años:
        Aux = Datos[Datos["fecha"].str.contains(str(Año))]
        Auxf = []
        Auxp = Datos[Datos["fecha"].str.contains(str(Año))]
        Auxp = Auxp.values.tolist()
        AuxA = []
        for i in range(len(Auxp)):
            if i ==0:
                AuxA.append(Auxp[i])
            else:
                if int(Auxp[i][0][-2:])-int(AuxA[0][0][-2:])<n_dias:
                    temp = max([Auxp[i][1],AuxA[0][1]])
                    AuxA[0][0] = Auxp[i][0]
                    AuxA[0][1] = temp
                else:
                    Auxf.append(AuxA)
                    AuxA = []
                    AuxA.append(Auxp[i])
        if len(AuxA)>0:
            Auxf.append(AuxA)
        for i in range(len(Auxf)):
            Datos_Independientes.append(Auxf[i][0][1])
    return Datos_Independientes

# Función que recibe una lista y comprueba si los datos en ella pasan el test de Anderson-Darling.
# En caso negativo, avanzan un intervalo (inter) y vuelven a comprobar hasta la convergencia.
# Devuelve la cantidad de datos en los que se llegó a la convergencia.

def Convergencia_AD(datos,T_inicial,inter=0.1,verb=False):
    Datos3 = datos.copy()
    T_Final = T_inicial
    Datos2  = Datos_Indep(Datos3)
    if verb:
        print("El threshold inicial es de {}".format(T_inicial))
        print("")
        
    while len(Datos2)>10:
        A = LBestPSO(Datos2,n_particulas=100,n_grupo=10)
        Ejemplo = A.fit(iter=1000)
        k = -1*Ejemplo[1]
        sigma = Ejemplo[2]
        A = A_2(Datos2,k,sigma)
        W = W_2(Datos2,k,sigma)
        test = Test_P_01(k,A,W)
        if verb:
            print("Anderson-Darling = {}".format(A))
            print("Cramer-Von Mises = {}".format(W))
        if test == "No se aprobó el test!":
            Minimo = min(Datos3["temint"])
            # Le resto el mínimo
            Datos3['temint'] = Datos3['temint'] - inter
            Datos3 = Datos3[Datos3['temint']>0]
            T_Final = round(T_Final + inter,2)
            Datos2  = Datos_Indep(Datos3)
            if verb:
                print("")
                print("============= No se convergió ====================")
                print("")
                print("================ El nuevo Threshold es {} =========".format(T_Final))
            

        else:
            if verb:
                print("Convergencia en {} datos".format(len(Datos3)))
                print("Quedo gamma = {} y Sigma = {}".format(-1*k,sigma))
            
            break
    if verb:
        print("El Threshold final quedó en {}".format(T_Final))
    print("Total de datos usados = {}".format(len(Datos2)))
    try:
        return -1*k,sigma,T_Final
    except:
        A = LBestPSO(Datos2,n_particulas=100,n_grupo=10)
        Ejemplo = A.fit(iter=1000)
        k = -1*Ejemplo[1]
        sigma = Ejemplo[2]
        return -1*k,sigma,T_Final