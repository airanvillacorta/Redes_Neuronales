import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0] 
def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",") # fichero iris.data
np.random.shuffle(data)
x_data = data[:, 0:4].astype('f4') # Datos. col 0,1,2,3
y_data = one_hot(data[:, 4].astype(int), 3) # Clase. col 4

print y_data

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

# Canal en la gpu donde entran/salen los datos.
# Canal de diametro 4 y longitud no se sabe. (4 propiedades flor iris, longitud num. elementos que introduciremos).
x = tf.placeholder("float", [None, 4]) # Entrada de datos gpu
y_ = tf.placeholder("float", [None, 3]) # Salida de datos gpu

# ---------

# W = tf.Variable(np.float32(np.random.rand(4, 3))*0.1)
# b = tf.Variable(np.float32(np.random.rand(3))*0.1)

# Primera capa, 5 neuronas 4 entradas
W = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1) # Matriz 4x5, 5 neuronas, 4 entradas. W son los pesos de cada neurona.
# Bias (Valor umbral expresado como peso sinaptico mas (-W0) asociado a neurona siempre activa).
b = tf.Variable(np.float32(np.random.rand(5)) * 0.1) # Valor aleatorio, no tenemos ningun criterio.

# y = tf.nn.softmax((tf.sigmoid(tf.matmul(x, W) + b)))

# Salida.
y = tf.sigmoid((tf.matmul(x, W) + b))  # Evitamos la normalizacion-> entrada * peso + b
# sigmoid-> Reduce el espacio a [0,1]
# W-> Todos los componentes de la red
# x-> Entradas
# b-> Bias

# ---------

# Segunda capa, 3 neuronas 5 entradas
Wi = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
bi = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

yi = tf.nn.softmax((tf.matmul(y, Wi) + bi))  # Normalizamos-> Calculamos yi = salida capa sup. * Peso capa act. + b
# softmax-> Normalizacion entre 0 y 1, fuerza a que la suma de como maximo 1, funcion de probabilidad para
# establecer la clase a la que pertenece.

# cross_entropy = tf.reduce_sum(tf.square(y_ - yi)) # Evitamos numeros negativos -> (y_ - yi)^2 (Restamos las salidas, elevamos al 2) -> Nos da el error
cross_entropy = -tf.reduce_sum(y_ * tf.log(yi))  # Reducimos la dimension reduce_sum() a 1 elemento.

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # Descenso por el gradiente para buscar el minimo.

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20 # Tamano del bloque. Introduciremos de 20 en 20.

vector = [] # Guardaremos las salidas para mostrarlo en la grafica.
for step in xrange(1000):
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]

        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys}) # Diccionario para alimentar el placeholder.
        if step % 50 == 0:

            e = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
            vector.append(e)
            print "Iteration #:", step, "Error: ", e
            result = sess.run(yi, feed_dict={x: batch_xs})
            for b, r in zip(batch_ys, result):
                print b, "-->", r
            print "----------------------------------------------------------------------------------"

# Mostramos la grafica
plt.plot(vector)
plt.xlabel('Iteration:')
plt.ylabel('Error:')
plt.show()

## No es separable linealmente con una sola neurona, por ello implementamos otra capa, con esto logramos poder separar
## los grupos y poder definir las clases. Con este metodo podemos clasificar elementos que no sean separables linealmente con
## una sola neurona. En esta practica-> (Usamos dos capas, una primera con 5 neuronas y una ultima con 3).



