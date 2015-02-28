"""
Module for reading in the signature database and normalizing it

@since: 23/02/15

@author: SCJ Robertson, 16579852
"""
import glob
import numpy as np
from matplotlib import pylab as plt

def load_signatures(dir_name="data/sign/*/*.txt", columns=[0, 1]):
	"""Return a list of raw signature data
	
	Return
	------
	data: (n,i,j) nd ndarray
	      n is the number of signatures
	      i,j the x,y coordinates of each signature
	"""
	data = []
	for fname in glob.glob(dir_name):
		data.append(np.genfromtxt(fname, comments='%')[:, columns].T)	
	return data

def remove_mean(signature_collection):
	"""
	Remove the mean position from a collection of signatures

	Parameters
	------
	signature_collection: (n,i,j) nd ndarray
			      n is the number of signatures
			      i,j the x,y coordinates of each signature

	Return
	-------
	zero_mean: (n,i,j) nd ndarray
		   n is the number of signatures
		   i,j the x,y coordinates of each signature
	"""
	data=signature_collection
	for d in data:
		X, Y=d[0, :], d[1, :]
		x_mean, y_mean=np.mean(X), np.mean(Y)
		X[:]=[x-x_mean for x in X[:]]
		Y[:]=[y-y_mean for y in Y[:]]
	return data
		
def rotate_signature(signature_collection):
	"""
	Rotate the a signature so its coordinate axes align with
	the coordinate axes. This also scales the signature so its
	largest singular value is one

	Parameters
	------
	signature_collection: (n,i,j) nd ndarray
			      n is the number of signatures
			      i,j the x,y coordinates of each signature
	Return
	-------
	rotated_signature: (n,i,j) nd ndarray
			   n is the number of signatures
			   i,j the x,y coordinates of each signature
		    sigma: (n) nd ndaray
			   Singular values of the signature presented as an array
		 rotation: (n,i,j) nd ndarray
		 	   n is the number of signatures
			   i,j are the dimensions of the U matrix
		 whitened: (n,i,j) nd ndarray
		 	   n is the number of signatures
			   i,j are the dimensions of the whitened data
	"""
	rotated_data=[]
	rotation=[]
	sigma=[]
	whitened=[]
	for D in signature_collection:
		U, s, V=np.linalg.svd(D, full_matrices=False)
		rotate_matrix(U)
		rotation.append(U)
		sigma.append(s)
		whitened.append(V)
		D_prime=np.dot(U.T, D)
		rotated_data.append(D_prime)	
	return rotated_data, rotation, sigma, whitened

def normalize_signature(signature_collection, S):
	"""
	Normalize a signature by reducing its largest singular value and variance to unity

	Parameters
	-------
	signature_collection: (n,i,j) nd ndarray
			      n is the number of signatures
			      i,j the x,y coordinates of each signature
			   S: (n,m) nd ndarray
			      Singular values of the respective signatures supplied as an array
	Returns
	--------
		  normalized: (n,i,j) nd ndarray
			      n is the number of signatures
			      i,j the x,y coordinates of each signature
	"""
	normalized=[]
	for d, s in zip (signature_collection, S):
		sigma_one=s[0]
		N=(d.shape)[1]
		scaling_factor=sigma_one/np.sqrt(N)
		normalized.append(d/scaling_factor)
	return normalized


def rotate_matrix(U):
	"""
	Flip the signature around the x,y axis

	Parameters
	-------
	signature_collection: (n,i,j) nd ndarray
			      n is the number of signatures
			      i,j the x,y coordinates of each signature
			   S: (n,m) nd ndarray
			      Singular values of the respective signatures supplied as an array
	Returns
	--------
		  	U: (i,j) nd ndarray
			    i,j the x,y coordinates of each signature
	"""
	if U[0, 0]<0:
		U[0, 0]=U[0, 0]*-1
	if U[1, 1]<0:	
		U[1, 1]=U[1, 1]*-1

def plot_signature(signature):
	"""
	Plot a single signature.

	Parameters
	-------
	signature: (i,j) nd ndarray
		   i,j the x,y coordinates of each signature
	"""
	plt.plot(signature[0, :], signature[1, :])
                                                                                                                         
def plot_ellipse(S):
	"""
	Plot an ellipse centered at zero for the given major and minor axes

	Parameters
	------
	S: (i,j) nd ndarray
	 i,j are the major and minor axes intercepts	
	"""
	t=np.linspace(0.0, 2*np.pi, 101)
	x, y =S[0]*np.cos(t), S[1]*np.sin(t)
	plt.plot(x, y)

def print_matrix(U):
	"""
	Print a given set of matrices

	Parameters
	-------
			U: (n,i,j) nd ndarray
			      n is the number of signatures
			      i,j the x,y coordinates of each signature
	"""
	for u in U:
		print u		

def main():
	"""Run some tests to see if the data has loaded"""
	n=0

	#This does the SVD and rotates D to correspond with the  principle directions
	signatures=load_signatures() 
	D=remove_mean(signatures)
	D_prime, U, S, V=rotate_signature(D) 
	
	#Scale the data so the largest singular and largest standard deviation are one
	D_scaled=normalize_signature(D_prime, S) 
	
	#Work out the minor and major ellipse radii, this is the ellipse of the signature's covariance matrix
	sigma_one=(S[n])[0]
	sigma_two=(S[n])[1]
	N=(D_prime[n].shape)[1]

	plt.figure()
	plot_ellipse([sigma_one/np.sqrt(N), sigma_two/np.sqrt(N)])
	plot_signature(D_prime[n])

	#Plot the data with the unity ellipse
	plt.figure()
	plot_ellipse([1, 1]) 
	for i in np.arange(n,n+14):
		plot_signature(D_scaled[i])
	plt.gca().set_aspect('equal')

	#Load the 5-dimensional data
	D_5=load_signatures(columns=[0, 1, 2, 3, 4])
	U, s, V=np.linalg.svd(D_5[n], full_matrices=False)
	
	#For a single 5 dimensional signature, the principle components are
	k=np.arange(0,5)
	plt.figure()
	plt.stem(k, s)
	
	#Reduce the dimensions from 5 to 2
	s[2:5] =0 #Throw away the principle components 3 to 5
	S_mat=np.diag(s) #Reconstruct the S matrix, this is still a 5x5 matrix
	D_reconstruct=np.dot(np.dot(U, S_mat), V)

	#Plot the reconstructed data
	plt.figure()
	plt.subplot('211')
	plot_signature(signatures[n])
	plt.subplot('212')
	plot_signature(D_reconstruct)
	plt.show()



if __name__=='__main__':
	try:
		main()
	except KeyboardInterrupt:
		exitapp=True

