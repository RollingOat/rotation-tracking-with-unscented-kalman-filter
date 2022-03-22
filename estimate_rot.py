# %%
from cProfile import label
from re import S
import numpy as np
from scipy import io
from quaternion import Quaternion
import os
# %%
def convert_accel(acceleration):
    '''
    Convert acceleration in digital readings to actual values in m/s^2
    INPUT:
        acceleration: (3, T) ndarray, digital readings
    OUTPUT:
        value: (3, T) ndarray, actual values in m/s^2
    '''
    
    Vref = 3300

    acc_x = -np.array(acceleration[0]) # IMU Ax and Ay direction is flipped !
    acc_y = -np.array(acceleration[1])
    acc_z = np.array(acceleration[2])
    acc = np.array([acc_x, acc_y, acc_z]).T

    acc_sensitivity = 33.86
    acc_scale_factor = Vref/1023.0/acc_sensitivity
    acc_bias = np.mean(acc[:20], axis=0) - np.array([0,0,9.81])/acc_scale_factor
    acc = (acc-acc_bias)*acc_scale_factor
    return acc.T

def convert_gyro(raw):
    '''
    Convert gyros in digital readings to actual values in mV/(radian/sec)
    INPUT:
        raw: (3, T) ndarray, digital readings of gyros
    OUTPUT:
        value: (3, T) ndarray, actual values in m/s^2
    '''
    
    Vref = 3300
    gyro_x = np.array(raw[1]) # angular rates are out of order !
    gyro_y = np.array(raw[2])
    gyro_z = np.array(raw[0])
    gyro = np.array([gyro_x, gyro_y, gyro_z]).T

    gyro_bias = np.mean(gyro[:20], axis=0)
    gyro_sensitivity = 193.55
    gyro_scale_factor = Vref/1023/gyro_sensitivity
    gyro = (gyro-gyro_bias)*gyro_scale_factor
    
    return gyro.T

def compute_W_i(P_last, Q): #correct
    '''
    INPUT: 
        n: the degree of freedom of the system, 6 in this case
        P_last: the covariance of last state, (n,n) ndarray
        Q: the covariance of noise in process model, (n,n) ndarray
    RETURN:
        W_i: (2*n, n) ndarray, a useful set of vectors in computing sigma points
    '''
    n = P_last.shape[0]
    S = np.linalg.cholesky(P_last+Q) # cholesky decomposition
    # print('S',S)
    W_i = np.zeros((2*n, n))
    for i in range(n):
        W_i[i] = np.sqrt(2*n) * S[:,i]
        W_i[i+n] = -W_i[i]
    return W_i
    
def compute_X_i(W_i, x_last): #correct
    '''
    INPUT:
        W_i: (2*n, n) ndarray, a useful set of vectors in computing sigma points
        x_last:(n+1,) ndarray, the mean of last state distribution with the first four elements representing quaternion
    RETURN:
        X_i: sigma points in n+1 dimension [quaternion, angular velocity], (2n,n+1) ndarray
    '''
    n = W_i.shape[1]
    X_i = np.zeros((2*n,n+1))
    for i in range(2*n):
        q = Quaternion()
        rotation_vector = W_i[i,0:3] # the axis-angle representation aka rotation vector
        q.from_axis_angle(rotation_vector)
        # print(np.linalg.norm(np.array([q.scalar(),q.vec()[0], q.vec()[1], q.vec()[2]])))
        q_W = q # the rotation component of W_i in quaternion form
        w_W = W_i[i, 3:6] # the angular velocity component of W_i 
        q_x_last = Quaternion(scalar= x_last[0], vec = [x_last[1], x_last[2], x_last[3]]) # the quaternion component of x_last
        w_x_last = x_last[4:7] # the augular velocity component of x_last

        q_X_i= q_x_last * q_W # calculate the quaternion component of the sigma point
        X_i[i,0:4] = np.array([
            q_X_i.scalar(), q_X_i.vec()[0], q_X_i.vec()[1], q_X_i.vec()[2]
        ])
        X_i[i,4:7] = w_x_last + w_W # calculate the angular velocity component of the sigma point
    
    return X_i

def compute_Y_i(X_i,t,obs,x_last): # correct
    '''
    INPUT:
        X_i: (2n, n+1) ndarray, sigma point of state_k
        t: time interval between each step
        x_last: (n+1, ) ndarray, previous state mean
    RETURN:
        Y_i: (2n, n+1) ndarray, the transformed sigma points by A(X_i,0), the process model 
    '''
    n = X_i.shape[1]-1
    Y_i = np.zeros((2*n,n+1))
    for i in range(2*n):
        w_k = x_last[4:7] # angular velocity in vector form
        # alpha_delta = np.linalg.norm(w_k) * t
        # axis_delta = w_k/np.linalg.norm(w_k)
        # q_delta = Quaternion(scalar=np.cos(alpha_delta/2), vec = axis_delta*np.sin(alpha_delta/2))
        q_delta = Quaternion()
        q_delta.from_axis_angle(w_k * t)
        q_k = Quaternion(scalar = X_i[i,0], vec = [X_i[i,1], X_i[i,2], X_i[i,3]]) # quaternion component of X_i[i]
        # q_k.normalize()
        q_x_next = q_k * q_delta # the quaternion component of prior state

        Y_i[i,0:4] = np.array([
            q_x_next.scalar(), q_x_next.vec()[0], q_x_next.vec()[1], q_x_next.vec()[2]
        ])

        Y_i[i,4:7] = X_i[i,4:7] # the w component of prior state

    return Y_i

def compute_prior_mean(Y_i,x_last): #correct
    '''
    INPUT:
        Y_i: (2n,n+1) ndarray, Y_i = A(X_i, 0)
        x_last: (n+1,) ndarray, the mean of last state distribution
    RETURN:
        x_bar: (n+1,) ndarray, the mean of Y_i
    '''
    n = Y_i.shape[1]-1
    # calculate mean of the augular velocity part 
    w_Y_i = Y_i[:,4:7]
    w_Y_bar = np.sum(w_Y_i,axis=0)/(2*n)

    # gradient descent algorithm to calculate the mean of quaternion
    q_Y_i = Y_i[:,0:4]

    current_q_mean = Quaternion(scalar=x_last[0], vec = list(x_last[1:4]))
    found = False # already find the true mean?
    mean_vector = [] # store q_bar for debug
    while not found:
        e_vector = np.zeros((2*n,3))
        for i in range(2*n):
            q = Quaternion(
                scalar= q_Y_i[i,0], vec= list(q_Y_i[i,1:4])
            ) # quaternion form of q_Y_i
            e_i = q * current_q_mean.inv() # quaternion form
            e_i.normalize()
            e_i_vector = e_i.axis_angle()   # rotation vector form
            if np.linalg.norm(e_i_vector) == 0: # not rotate
                e_i_vector = np.zeros(3)
            else:
                e_i_vector = (-np.pi + np.mod(np.linalg.norm(e_i_vector) + np.pi, 2 * np.pi)) / np.linalg.norm(e_i_vector) * e_i_vector
            
            e_vector[i] = e_i_vector
            

        e_vector_mean = np.mean(e_vector, axis = 0)

        # store e_vector_mean for debug
        mean_vector.append(np.linalg.norm(e_vector_mean))
        e_q_mean = Quaternion()
        e_q_mean.from_axis_angle(e_vector_mean) # quaternion form of e_vector_mean
        current_q_mean = e_q_mean * current_q_mean # update q_mean
        current_q_mean.normalize()
        tol = 1e-1 # set tolerance of comparison
        if np.linalg.norm(e_vector_mean) < tol:
            q_mean = current_q_mean # quaternion form of e_vector_mean
            found = True
        
            
    # plot e_vector_mean for debug
    # fig,ax = plt.subplots()
    # ax.plot(mean_vector)
    e_vector_last_iteration = e_vector
    # q_mean, error = quat_average(q_Y_i, x_last[0:4])
    x_bar = np.zeros(n+1)
    x_bar[0] = q_mean.scalar()
    x_bar[1:4] = q_mean.vec()
    x_bar[4:7] = w_Y_bar

    return x_bar, e_vector_last_iteration

def compute_W_prime_i(Y_i,x_bar,e_last_iteration): #correct
    '''
    INPUT: 
        Y_i: (2n,n+1) ndarray, Y_i = A(X_i, 0)
        x_bar: (n+1,) ndarray, the mean of Y_i
    RETURN:
        W_prime_i: (2n,n) ndarray, a useful parameter in computing covariance matrix of prior state
    '''
    n = len(x_bar)-1
    # convert the quaternion part to rotation vector
    w_W_mean_removed = (Y_i - x_bar)[:,4:7] # the angular velocity part
    W_prime_i = np.zeros((2*n,n)) 
    for i in range(2*n):
        W_prime_i[i,0:3] = e_last_iteration[i]
        W_prime_i[i,3:6] = w_W_mean_removed[i]
    return W_prime_i

def compute_prior_covariance(W_prime_i):
    '''
    INPUT:
        W_prime_i: (2n,n) ndarray, a useful parameter in computing covariance matrix of prior state
    RETURN:
        P_bar: (n,n) ndarray,covariance matrix of prior state x_k+1_k
    '''
    n = W_prime_i.shape[1]
    P_bar = np.zeros((n,n))
    for i in range(2*n):
        W = W_prime_i[i].reshape(-1,1) # the row vector of W_prime_i
        W_WT = W @ W.T # (n,1) @ (1,n) = (n,n)
        P_bar = P_bar + W_WT # sum of W_WT
    
    P_bar = P_bar/(2*n) # mean of the sum

    return P_bar


def compute_Z_i(Y_i):
    '''
    INPUT:
        Y_i: (2n,n+1) ndarray, Y_i = A(X_i, 0)
    RETURN:
        Z_i: (2n,n) ndarray, the observation vector obtained from Z = H(Y,0) 
    '''
    n = Y_i.shape[1]-1
    Z_i = np.zeros((2*n,n))
    # the angular velocity part of Z_i
    Z_i[:,3:6] = Y_i[:,4:7]

    # the rotation vector part
    for i in range(2*n):
        g_vector = [0, 0, 9.81]
        g_quaternion = Quaternion(scalar=0, vec = g_vector)
        q_k = Quaternion(scalar=Y_i[i,0], vec=list(Y_i[i,1:4]))
        g_prime_quaternion = q_k.inv() * g_quaternion * q_k # result is a pure quaternion
        g_prime_vector_part = g_prime_quaternion.vec() # I use the vector part of the quaternion here, not the axis-angle representaion
        Z_i[i,0:3] = g_prime_vector_part
    
    return Z_i

def compute_measurement_mean(Z_i):
    '''
    INPUT:
        Z_i: (2n,n) ndarray, the observation vector obtained from Z = H(Y,0) 
        z_true: (n,) ndarray, the actual observation [acceleration, angular velocity]
    OUTPUT:
        z_bar: (n, ) ndarray, mean of Z_i
        v: (n,) ndarray, innovation v = z - z_bar
    '''
    n = Z_i.shape[1]
    z_bar = np.sum(Z_i,axis = 0)/(2*n)
    return z_bar

def compute_innovation(z_bar,z_true):
    '''
    INPUT:
        z_true: (n,) ndarray, the actual observation [acceleration, angular velocity]
    OUTPUT:
        v: (n,) ndarray, innovation v = z - z_bar
    '''
    v = z_true - z_bar
    return v

def compute_measurement_covariance(R,z_bar,Z_i,W_prime_i):
    '''
    INPUT:
        R: (n,n) ndarray, measurement noise covariance
        z_bar: (n,) ndarray, mean of Z_i
        Z_i: (2n,n) ndarray
    OUTPUT:
        P_zz: (n,n) ndarray, covariance of measurement vector
        P_vv: (n,n) ndarray, covariance of innovation vector
        P_xz; (n,n) ndarray, cross correlation matrix of Y_i and Z_i
    '''
    n = len(z_bar)
    P_zz = np.zeros((n,n))
    P_vv = np.zeros((n,n))
    P_xz = np.zeros((n,n))
    for i in range(2*n):
        row_vector_z = (Z_i[i] - z_bar).reshape(-1,1) # row vector of Z_i - z_bar
        P_zz = P_zz + row_vector_z @ row_vector_z.T
        P_xz = P_xz + W_prime_i[i].reshape(-1,1) @ row_vector_z.T # (n,1)(1,n) = (n,n)
    
    P_zz = P_zz/(2*n)
    P_xz = P_xz/(2*n)
    P_vv = P_zz + R

    return P_zz, P_xz, P_vv

def compute_kalman_gain(P_xz, P_vv):
    '''
    INPUT:
        P_vv: (n,n) ndarray, covariance of innovation vector
        P_xz; (n,n) ndarray, cross correlation matrix of Y_i and Z_i
    OUTPUT:
        K: (n,n) ndarray, kalman gain of undecented kalman filter
    '''
    K = P_xz @ np.linalg.inv(P_vv)
    return K

def compute_posterior_mean(x_bar, K, v):
    '''
    INPUTS:
       x_bar: (n+1,) ndarray, the mean of Y_i
       K: (n,n) ndarray, kalman gain of undecented kalman filter
       v: (n,) ndarray, innovation v = z - z_bar
    RETURN:
        mean: (n+1,) ndarray, the posterior mean or updated mean [quaternion, angular velocity]
    '''
    n = len(x_bar)-1
    q_x_bar = Quaternion(scalar=x_bar[0], vec = list(x_bar[1:4])) # quaternion part of x_bar
    
    # convert K @ v into quaternion
    kv = K @ v.reshape(-1,1)
    
    kv = kv.flatten()
    q_kv = Quaternion()
    q_kv.from_axis_angle(kv[0:3])
    q_mean = q_kv * q_x_bar  # quaternion part of the mean
    w_mean = kv[3:6] + x_bar[4:7]
    mean = np.zeros(n+1)
    mean[0:4] = np.array([q_mean.scalar(), q_mean.vec()[0], q_mean.vec()[1], q_mean.vec()[2]])
    mean[4:7] = w_mean

    return mean

def compute_posterior_covariance(K, P_bar, P_vv):
    '''
    INPUTS:
        K: (n,n) ndarray, kalman gain of undecented kalman filter
        P_bar: (n,n) ndarray,covariance matrix of prior state x_k+1_k
        P_vv: (n,n) ndarray, covariance of innovation vector
    OUTPUT:
        covariance: (n,n) ndarray, the updated covariance
    '''
    covariance = P_bar - K @ P_vv @ K.T

    return covariance

def update(x_last, P_last, Q, R, t, z_true,obs_last):
    '''
    One step update
    INPUT:
        x_last: (n+1,) ndarray, the mean of last state distribution at step k-1
        P_last: (n,n) ndarray, the covariance of last state distribution at step k-1
        Q: (n,n) ndarray, the state noise covariance
        R: (n,n) ndarray, the measurement noise covariance
        t: time interval between two steps
        z_true: (n,) ndarray, observation at step k
    RETURN:
        x: (n+1, ), updated mean
        P: (n,n), updated variance
    '''
    W_i = compute_W_i(P_last,Q)
    X_i = compute_X_i(W_i,x_last)
    Y_i = compute_Y_i(X_i,t,obs_last,x_last)
    x_bar, e_vector_last_iteration = compute_prior_mean(Y_i,x_last)
    W_prime_i = compute_W_prime_i(Y_i,x_bar,e_vector_last_iteration)
    P_bar = compute_prior_covariance(W_prime_i)
    Z_i = compute_Z_i(Y_i)
    z_bar = compute_measurement_mean(Z_i)
    v = compute_innovation(z_bar,z_true)
    P_zz, P_xz, P_vv = compute_measurement_covariance(R, z_bar, Z_i, W_prime_i)
    K = compute_kalman_gain(P_xz, P_vv)
    x = compute_posterior_mean(x_bar, K, v)
    P = compute_posterior_covariance(K, P_bar, P_vv)
    
    return x,P

# %%
def estimate_rot(data_num = 1):
    filename = os.path.join(os.path.dirname(__file__),"imu/imuRaw" + str(data_num) + ".mat")
    imu = io.loadmat(filename) 

    # vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

    # your code goes here
    # vicon_rot_m = vicon['rots']
    accel_values = convert_accel(accel)
    gyro_values = convert_gyro(gyro)
    observation = np.vstack((accel_values, gyro_values))

    # initialize some parameters R, Q, x_0, P_0
    time = imu['ts'].flatten()
    # time_vicon = vicon['ts'].flatten()
    Q = np.diag([3]*6) # PROCESS MODEL NOISE
    R = np.diag([3]*6)  # OBSERVATION MODEL NOISE

    # initialize x_0
    x_0 = np.zeros(7)
    x_0[0:4] = np.array([1, 0, 0, 0])
    x_0[4:7] = np.array([0, 0, 0]) # gyro_values[:,0]

    # initialize P_0
    P_0 = np.diag([1]*6)
    x = x_0
    P = P_0

    # roll, pitch, yaw are numpy arrays of length T
    roll = np.zeros(T)
    pitch = np.zeros(T)
    yaw = np.zeros(T)
    # euler_angle_0 = q_0.euler_angles()
    roll[0] = 0
    pitch[0] = 0
    yaw[0] = 0

    # store x in the process
    state = np.zeros((T,7)) # estimated mean
    state[0] = x_0

    # start filter process
    for i in range(1,T):
        t = time[i]-time[i-1]
        x, P = update(x, P, Q, R, t, observation[:,i],observation[:,i-1])
        state[i] = x
        q = Quaternion(scalar = x[0], vec = list(x[1:4]))
        euler_angle = q.euler_angles()
        roll[i] = euler_angle[0]
        pitch[i] = euler_angle[1]
        yaw[i] = euler_angle[2]

    return roll, pitch , yaw

estimate_rot(1)

