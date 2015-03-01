from sklearn.decomposition import PCA
from matplotlib import pylab as plt
import numpy as np
import os.path
import glob

def load_file(sign_data):
	sign_data = np.loadtxt(sign_data, comments='%')
	sign_data = sign_data[:,:2]
	return sign_data

def zero_mean(sign_data):
	sign_batch = sign_data

	sign_batch = sign_data[:,:2]

	mean_x = np.mean(sign_batch[:,0])
	mean_y = np.mean(sign_batch[:,1])

	sign_data[:,0] = sign_batch[:,0] - mean_x
	sign_data[:,1] = sign_batch[:,1] - mean_y
  	return sign_batch
"""
def rotate_data(U, sign_data):
	if U[0][0] < 0:
    U[0][0] = -1*U[0][0]
        
	if U[1][1] < 0:
    U[1][1] = -1*U[1][1]

	sign_rotated = np.dot(U.T, sign_data.T)
	return sign_data
"""
def scale(V, sign_data):
	N = len(sign_data)
	scale_factor = np.sqrt(N)
	V = np.dot(V,scale_factor)
	return V

def rotate_mat(U,V):
	if U[0][0] < 0:
	 V[0,:] = -1 * V[0,:]
    	 U[0][0] = -1 * U[0][0]
   	 U[1][0] = -1 * U[1][0]
	if U[1][1] < 0:
   	 U[1][1] = -1 * U[1][1]
    	 U[0][1] = -1 * U[0][1]
	return U, V

def norm(S, sign_data):
	norm_factor = S[0]
 	S = S/S[0]
 	sign_data = np.dot(sign_data,norm_factor)
 	return S_new, sign_data

def read_batch(location):
	plt.figure()
	plt.xlabel('X-Component')
	plt.ylabel('Y-Component')
	plt.title('All signautres')
	for sign_data in os.listdir(location):
		sign_data = load_file(location + sign_data) 
		sign_data = zero_mean(sign_data)
		U, S, V =np.linalg.svd(sign_data.T, full_matrices=False)
		U, V = rotate_mat(U,V)    	
		V_norm = scale(V, sign_data)
		plt.plot(V_norm[0,:],V_norm[1,:])
		plt.axis('equal')
	plt.show()
	
