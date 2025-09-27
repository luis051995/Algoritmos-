import numpy as np
import pandas as pd

#1 Datos: Colores  basicos  
data = pd.DateFrame({
    'R':[1,0,0,0,0,1],
    'G':[0,1,0,1,1,0],
    'B':[0,0,1,0,1,1],
    'Label': {'Rojo','Verde','Azul','Amarillo','Cian','Magenta'}
})

#ONE-HOT ENCODING
color_map = {
    'Rojo': [1,0,0],
    'Verde': [0,1,0],
    'Azul': [0,0,1],   
    'Amarillo': [1,1,0],
    'Cian': [0,1,1],
    'Magenta': [1,0,1]
    }

X=data[['R','G','B']].values
y=np.array([color_map[label] for label in data['Label']])

#Nomalizacion datos 
X = X.asytrpe(float)

#Parametros de la red neuronal
n_input = 3
n_hidden = 4
n_output = 3

np.random.seed(42)
W1 = np.random.rand(n_input, n_hidden)*0.1
W2 = np.random.rand(n_hidden, n_output)*0.1
b1 = np.zeros((1, n_hidden))
b2 = np.zeros((1, n_output))

#Funcion activacion
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)  
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#3 Entrenamiento
lr = 0.1
n_epochs = 10000

for epoch in range(n_epochs):
   
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = softmax(final_input)
   
    loss = -np.mean(np.sum(y * np.log(final_output + 1e-9), axis=1))
    
  
    error_output = final_output - y
    dW2 = np.dot(hidden_output.T, error_output) / X.shape[0]
    db2 = np.sum(error_output, axis=0, keepdims=True) / X.shape[0]
    
    error_hidden = np.dot(error_output, W2.T) * sigmoid_derivative(hidden_output)
    dW1 = np.dot(X.T, error_hidden) / X.shape[0]
    db1 = np.sum(error_hidden, axis=0, keepdims=True) / X.shape[0]
    
    #Actualizar pesos y sesgos
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
        
#prdiccion 
def predict(X):
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = softmax(final_input)
    return np.argmax(final_output, axis=1)
predictions = predict(X)
print("Predicciones:", predictions)