from mitdeeplearning import util
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Flatten
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model

from IPython.display import clear_output

import ot  
import ot.plot


__all__ = [
    "Shannon_network",
]

# Neural nerwork architecture
class Shannon_network:
    def __init__(self, C, eps):
        self.C = C
        self.eps = eps
        self.K = np.exp(-C/C.max()/eps)
        self.n_marginals = len(C.shape)
        self.shapes = (self.n_marginals, C.shape[0], )
        
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    
    def build_model(self): #model via functional api 
        input_msrs = Input(shape=self.shapes, name="Marginals") # takes array of marginal measures
        flat_msrs = Flatten(name="Marginals_flat")(input_msrs) # flattens the marginals into 1 array
        log_msrs = Lambda(lambda x: tf.math.log(tf.cast(x, tf.float32)), name="logM")(flat_msrs) # additional input log
        new_inputs = concatenate([flat_msrs, log_msrs], name="M_concat_logM") # makes a new kernel that has MU, log(MU)
        
        hidden1 = Dense(20, activation='relu', name="hidden1", kernel_initializer='zeros')(new_inputs) 
        
        output_potentials = [Dense(self.shapes[1], name=f"u{i}", kernel_initializer='zeros', 
                         bias_initializer='zeros')(hidden1) for i in range(1, self.n_marginals)] # outputs N-1 predicted U
        
        model = Model(inputs=[input_msrs], outputs=output_potentials) #stacks all layers together
        return model
    
    @staticmethod
    def plot_P(mu, nu, P): # plots estimated Coupling projected on first 2 axes
        plt.figure(figsize=(7,7))
        ot.plot.plot1D_mat(mu, nu, P.sum(axis=tuple([i for i in range(2, len(P.shape))])))
        return plt
        
    def train_step(self, marginals): #optimization step
        potentials = self.predict_potentials(marginals)
        P = self.predict_P(potentials)
        with tf.GradientTape() as tape: #assign loss function
            loss = self.loss(marginals)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, potentials, P
        
    def fit(self, marginals, epochs=100): # model train
        plotter = util.PeriodicPlotter(sec=1, xlabel='Iterations', ylabel='Loss')
        if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

        for iter in tqdm(range(epochs)): #plotting the steps of network updates
            loss, potentials, P = self.train_step(marginals)
            # print(loss)
            clear_output(wait=True)
            fig = self.plot_P(marginals[0, 0], marginals[0,1], P[0].numpy())
#             fig.savefig(f"C2\\dnn\\N={self.n_marginals}\\{iter}.png")
            fig.show()
        
        return potentials, P.numpy()
    
    def U_call(self, marginals): #estimate u_1, ..., u_n-1 
        prediction = self.model.call({"Marginals": tf.Variable(marginals)})   
        return prediction if type(prediction)== list else [prediction] 
    
    def call(self, marginals): # predict P(marginals)
        potentials = self.predict_potentials(marginals)
        return self.predict_P(potentials).numpy()
    
    def to_tensors(self, vec, pos): # reshapes vectors to do elementwise operations with tensors for all samples
        shape = [1]*(1+self.n_marginals)
        n_samples, n = vec.numpy().shape
        
        shape[0] = n_samples
        shape[pos+1] = n
        return tf.reshape(vec, tuple(shape))
    
    @property
    def K_reshaped(self): # reshapes exp(-Cost) tensor to do operations for all samples
        return tf.reshape(self.K, (1,)+self.K.shape)
    
    def dual(self, marginals): # entropic dual functional for sample
        U = self.predict_potentials(marginals)
        P = self.predict_P(U)
        D = - self.eps*tf.reduce_sum(P, axis=range(1,self.n_marginals+1))
        for i, u in enumerate(U):
            D += tf.reduce_sum(u*marginals[:, i], axis=1)
        return D 

    def loss(self, marginals):
        return -tf.reduce_sum(self.dual(marginals))
    
    def predict_potentials(self, marginals): # estimate all potentials 
        U = self.U_call(marginals)
        factor = self.K_reshaped
        potentials = []
        
        # estimate u_n from u_1... u_n-1
        for i, u in enumerate(U):
            potentials.append(u)
            factor = factor * self.to_tensors(tf.math.exp(u/self.eps), i)
            
        u_n = self.eps*(tf.math.log(marginals[:, -1]) - tf.math.log(tf.reduce_sum(factor, axis=range(1, self.n_marginals))))
        potentials.append(u_n)
        return potentials 
    
    def predict_P(self, potentials): # estimate P(potentials)
        factor = self.K_reshaped
        for i, u in enumerate(potentials):
            factor = factor * self.to_tensors(tf.math.exp(u/self.eps), i)
        return factor
    
    @property
    def plot_model(self): #plots the netrork structure
        return plot_model(self.model, show_shapes=True)
    
    @staticmethod
    def plot_cost(C): # plots Cost tensor projected on first 2 axes
        plt.figure(figsize=(7,7))
        plt.imshow(C.sum(axis=tuple([i for i in range(2, len(C.shape))])))
        plt.title("Cost function")
        return plt.show()
        
        