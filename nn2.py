import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy
import time
from line_profiler import LineProfiler


DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

#solution exactes
data = scipy.io.loadmat('/home/celhage/Documents/ei2/Projet8/myPINN/data/burgers_shock.mat')
u_vrai = data['usol']

#constantes
pi = tf.constant(np.pi, dtype=DTYPE)
viscosity = .01/pi

class PINN(tf.keras.Model):
    """ 
    Définie l'architecture du PINN
    """

    def __init__(self, lb, ub, 
            output_dim=1,
            num_hidden_layers=8, 
            num_neurons_per_layer=20,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb  
        self.ub = ub
        
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim, 
                                         activation=tf.keras.activations.get(activation))


    @tf.function
    def call(self, X, training = False):
        '''
        Avance forward dans le NN
        '''
        Y = self.scale(X)
        for i in range(self.num_hidden_layers):
            Y = self.hidden[i](Y, training = True)
        output = self.out(Y)
        return(output)
    


class PINNSolver():
    '''
    Définie un solveur en se basant sur un modele déjà défini
    '''

    def __init__(self, model, X_r, gamma = 1, batch_size = 32):
        self.model = model
        self.batch_size = batch_size
        
        self.t = X_r[:,0:1]
        self.x = X_r[:,1:2]

        # historique de loss et nmbre d'iterations
        self.hist_loss = []
        self.epoch = 0
        self.time_epoch = 0
        self.batch = 0
        self.time_batch = 0
        self.gamma = gamma


    def f_r(self, t, x, u, u_t, u_x, u_xx):
        '''
        Partie résiduelle de l'EDP
        '''
        return u_t + u * u_x - viscosity * u_xx
        

    def get_r(self):
        '''
        Calcule le residu de l'EDP
        '''
        batch_t, batch_x = self.iterator.get_next()

        with tf.GradientTape(persistent = True) as tape:
            tape.watch(batch_t)
            tape.watch(batch_x)                
            
            u = self.model(tf.stack([batch_t[:,0], batch_x[:,0]], axis=1), training = True)
                
            u_x = tape.gradient(u, batch_x)
            u_t = tape.gradient(u, batch_t)
            u_xx = tape.gradient(u_x, batch_x)

        del tape

        return self.f_r(batch_t, batch_x, u, u_t, u_x, u_xx)
    

    def loss_f(self, X, u):
        '''
        Calcule de loss
        '''
        r = self.get_r()
        MSEf = tf.reduce_mean(tf.square(r)) 
                
        u_pred = self.model(X, training = True)
        MSEu = tf.reduce_mean(tf.square(tf.subtract(u_pred, u)))
            
        loss = self.gamma*MSEf + MSEu

        return loss
    

    @tf.function
    def get_grad(self, X, u):
        '''
        Calcule le gradient par rapport aux poids et aux biais
        '''
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.loss_f(X, u)
            
            grad = tape.gradient(loss, self.model.trainable_variables)
                
        return loss, grad
    

    def create_batches(self, N):
        combined_tensor = tf.concat([self.t, self.x], axis=-1)
        shuffled_combined_tensor = tf.random.shuffle(combined_tensor)

        t_shuf = shuffled_combined_tensor[:, :self.t.shape[-1]]
        x_shuf = shuffled_combined_tensor[:, self.t.shape[-1]:]
    
        batches = tf.data.Dataset.from_tensor_slices((t_shuf, x_shuf)).batch(self.batch_size).repeat(N)
        self.iterator = iter(batches)
        return(int(len(batches)/N))

    
    
    def solve(self, optimizer, X, u, N=1001):
        '''
        optimisation par SGD
        '''
        @tf.function
        def train_step():
            '''
            Réalise un pas de la SGD
            '''
            loss, grad = self.get_grad(X, u)
            
            optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
            return loss
        
        nbr_batches = self.create_batches(N)  
        for i in range(N * nbr_batches):

            start_time = time.time()
            loss = train_step()
            end_time = time.time()
            self.time_batch = end_time - start_time

            self.current_loss = loss.numpy()
            self.callback_batch()

            if (i+1) % nbr_batches == 0:
                self.callback_epoch()

    
    def callback_batch(self):
        '''
        Print nombre de batch dans l'epoch, la valeur de loss et temps de calcul
        '''
        self.time_epoch += self.time_batch
        print('It_epoch {:05d}, time_epoch = {:10.4e} s \
              It_batch {:05d}, time_batch = {:10.4e} s, \
              loss = {:10.4e}'.format(self.epoch, self.time_epoch,
                                      self.batch, self.time_batch, 
                                      self.current_loss))
        self.batch += 1



    def callback_epoch(self):
        '''
        Print le nombre d'epoch, la valeur de loss et temps de calcul depuis derniere epoch
        '''
        print('It_epoch {:05d}, time_epoch =  {:10.4e} ,loss = {:10.8e}'.format(self.epoch, self.time_epoch, self.current_loss))
        self.hist_loss.append(self.current_loss)
        self.epoch+=1
        self.time_epoch = 0
        self.batch = 0
        

    def data_prep(self, N_x = 1000, N_t = 1000):
        '''
        Preparation des données pour les plots et la metrique 
        '''
        tspace = np.linspace(self.model.lb[0], self.model.ub[0], N_t+1)
        xspace = np.linspace(self.model.lb[1], self.model.ub[1], N_x+1)

        T, X = np.meshgrid(tspace, xspace)
        Xgrid = np.vstack([T.flatten(),X.flatten()]).T 
        upred = self.model(tf.cast(Xgrid,DTYPE))
        U = upred.numpy().reshape(N_x+1,N_t+1)
        return(tspace,xspace,U)
    

    def metrique(self):
        '''
        Calcule la racine de l'erreur quadratique réduite entre la prédiction du modèle et la solution exacte 
        '''
        tspace, xspace, U = self.data_prep(N_x=255, N_t=99)
        metrique = np.sqrt(np.mean((U-u_vrai)**2))
        return(metrique)


    def plot_solution(self, nom_save = 'plot_model'):
        '''
        Plot le graphique 
        '''
        tspace, xspace, U = self.data_prep()

        #creation fig et ax + tracé
        fig, ax = plt.subplots()
        fig.set_size_inches(19,12)
        title_style = dict(pad=10, fontname="Ubuntu", fontsize=25)
        img = ax.pcolormesh(tspace, xspace, U, shading="auto", cmap = 'plasma')

        #ajout d'une barre d'echelle
        bar = plt.colorbar(img, ax=ax)

        #ajout label et titre
        ax.set_xlabel('t', fontsize = 20)
        ax.set_ylabel('x', fontsize = 20)
        ax.set_title('u', **title_style)
        plt.savefig("plot/"+nom_save)


    def plot_25_50_75(self, nom_save = 'plot'):
        '''
        Plot 3 graphiques comparant solutions exactes et solution prédites
        '''
        tspace, xspace, U = self.data_prep(N_t = 99, N_x = 255)

        u_t_25=U[:,25]
        u_t_50=U[:,50]
        u_t_75=U[:,25]

        fig, axs = plt.subplots(1,3, figsize=(20,15), sharey=True)
        title_style2 = dict(pad=10, fontname="Ubuntu", fontsize=18)
        axs[0].set_ylabel('u(x,t)', fontsize = 17)

        l1,=axs[0].plot(xspace, u_vrai[:,25], linewidth=6, color='b')
        l2,=axs[0].plot(xspace,u_t_25,linewidth=6,linestyle='dashed',color='r')
        axs[0].set_title('t=0.25', **title_style2)
        axs[0].set_xlabel('x', fontsize=15)

        axs[1].plot(xspace, u_vrai[:,50], linewidth=6, color='b')
        axs[1].plot(xspace, u_t_50, linewidth=6, linestyle='dashed', color='r')
        axs[1].set_title('t=0.50', **title_style2)
        axs[1].set_xlabel('x', fontsize = 15)

        axs[2].plot(xspace, u_vrai[:,75], linewidth=6, color='b')
        axs[2].plot(xspace, u_t_75, linewidth=6, linestyle='dashed', color='r')
        axs[2].set_title('t=0.75', **title_style2)
        axs[2].set_xlabel('x', fontsize = 15)
        
        fig.legend(handles=(l1,l2),labels=('Exact','Predicted'),loc='upper right', fontsize = 15)
        plt.savefig("plot/25_50_75_"+nom_save)
        return()
        

    def plot_loss_history(self, nom_save='plot_hist_loss'):
        '''
        Plot l'historique de la fonction loss
        '''
        fig = plt.figure(figsize=(20,15))
        ax = fig.add_subplot(111)
        

        ax.semilogy(range(len(self.hist_loss)), self.hist_loss,'k-') 

        title_style = dict(pad=10, fontname="Ubuntu", fontsize=18)
        ax.set_xlabel('$n_{epochs}$', fontsize = 15)
        ax.set_ylabel('loss', fontsize = 15)
        ax.set_title('Evolution de loss', **title_style)

        plt.savefig("plot/loss_history_"+nom_save)
        return()



# Constantes
pi = tf.constant(np.pi, dtype=DTYPE)
viscosity = .01/pi

# Fonction conditions initiales
def fun_u_0(x): 
    return -tf.sin(pi * x)

# Fonction conditions limites
def fun_u_b(t, x):
    n = x.shape[0]
    return tf.zeros((n,1), dtype=DTYPE)

# Nombre de train points
N_0 = 500
N_b = 500
N_r = 10000

# Limites du domaines
tmin = 0.
tmax = 1.
xmin = -1.
xmax = 1.

# Limite inf
lb = tf.constant([tmin, xmin], dtype=DTYPE)
# Limite sup
ub = tf.constant([tmax, xmax], dtype=DTYPE)


# Points initiales
t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)
X_0 = tf.concat([t_0, x_0], axis=1)

# Calcule condition initiales
u_0 = fun_u_0(x_0)

# Points limites 
t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
X_b = tf.concat([t_b, x_b], axis=1)

# Calcule condition limites
u_b = fun_u_b(t_b, x_b)

# Points dans le domaine
t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
X_r = tf.concat([t_r, x_r], axis=1)
shape = tf.shape(X_r).numpy()
u_r = tf.zeros(shape=(shape[0], shape[1]))

# Données pour condition limites et initiales
X_data = [X_0, X_b]
u_data = [u_0, u_b]


##plot des points

def plot_train_points(t_0, t_b, x_0, x_b,
                      t_r, x_r, u_b, u_0, 
                      title = 'Position des train points',
                      nom_save = 'train_points'): 
    fig, ax = plt.subplots()
    fig.set_size_inches(9,6)
    title_style = dict(pad=10, fontname="Ubuntu", fontsize=18)

    plt.scatter(t_0, x_0, c=u_0, marker='X', vmin=-1, vmax=1)
    plt.scatter(t_b, x_b, c=u_b, marker='X', vmin=-1, vmax=1)
    plt.scatter(t_r, x_r, c='r', marker='.', alpha=0.1)
    plt.xlabel('$t$')
    plt.ylabel('$x$')

    plt.title(title, **title_style)
    plt.savefig('plot/' + nom_save)
    return()

plot_train_points(t_0, t_b, x_0, x_b, t_r, x_r, u_b, u_0)

##fonction solve 

def solve_burger(num_model,
                 num_hidden_layers = 8, 
                 num_neurons_per_layer = 20, 
                 activation = 'tanh',
                 lb = lb,
                 ub = ub,
                 gamma = 1,
                 le_ra = ([100,300,400], [1e-3,5e-4,1e-4,5e-5]),
                 N = 501,
                 batch_size = 100):
    
    model = PINN(lb = lb, ub=ub,
                 num_hidden_layers=num_hidden_layers, 
                 num_neurons_per_layer=num_neurons_per_layer, 
                 activation=activation)
    model.build(input_shape=(None,2))

    solver = PINNSolver(model, X_r, gamma, batch_size)

    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(le_ra[0],le_ra[1])
    optim = tf.keras.optimizers.Adam(learning_rate=lr)

    temps_depart = time.time()
    solver.solve(optim, X_data, u_data, N = N)
    temps_final = time.time()
    temps_ecoule = temps_final - temps_depart

    model.save_weights('weights/model'+str(num_model)+'/weights')
    solver.plot_solution(nom_save='model'+str(num_model))
    solver.plot_25_50_75(nom_save='model'+str(num_model))
    solver.plot_loss_history(nom_save='loss_hist'+str(num_model))
    metrique = solver.metrique()
    print(f'metrique{num_model} = {metrique}')
    print(f'time{num_model} = {temps_ecoule} secondes')
    return(metrique, temps_ecoule)




#solve_burger(0, N = 11)

lp = LineProfiler()
lp_wrapper = lp(solve_burger)
lp_wrapper(0, N=11)
lp.print_stats()

#solve_burger(2, gamma = 0.25)
# solve_burger(3, gamma = 0.5)
# solve_burger(4, gamma=1.5)
# solve_burger(5, gamma=2)
# solve_burger(6, num_hidden_layers=2, num_neurons_per_layer=200)
# solve_burger(7, N = 1001, le_ra = ([100, 300, 500, 700, 900], [1e-3,1e-4,5e-5, 1e-5, 5e-6, 1e-6]))