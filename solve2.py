from nn import PINN, PINNSolver
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import xlwt

DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

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
N_r = 50000

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


#solve_burger(1)
solve_burger(2, batch_size=1000)
# solve_burger(3, gamma = 0.5)
# solve_burger(4, gamma=1.5)
# solve_burger(5, gamma=2)
# solve_burger(6, num_hidden_layers=2, num_neurons_per_layer=200)
# solve_burger(7, N = 1001, le_ra = ([100, 300, 500, 700, 900], [1e-3,1e-4,5e-5, 1e-5, 5e-6, 1e-6]))