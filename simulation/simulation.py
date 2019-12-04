import numpy as np
import math
import cv2
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt 


# a rank r svd projection
def svd_r_proj(A, r):
	U, s, Vt = np.linalg.svd(A, full_matrices=False) #get reduced SVD
	A_r = np.dot(U[:,:r]*s[:r],Vt[:r,:])
	return A_r
	
'''closed form solution for E|A_hat - A|_F^2 s.t. rank(A_hat)<=r'''
def closed_form(A, n_sigma2, rank):
	m, n = A.shape
	At_A = A.T @ A
	
	U, s, Vt = np.linalg.svd(A,full_matrices=False)
	s = np.sqrt(s*s+n_sigma2*m)
	M = Vt.T * s @ Vt
	M_inv = LA.pinv(M)
	B = M_inv @ At_A
	X = M_inv @ svd_r_proj(B, rank)
	
	return X

def approximated_form(C, n_sigma2, rank):
	m, n = C.shape
	I = np.ones((n,n))
	Ct_C = C.T @ C
	U, s, Vt = np.linalg.svd(C, full_matrices=False)
	Ct_CHalf = Vt.T * s @ Vt
	Ct_CHalf_inv = LA.pinv(Ct_CHalf)
	B = Ct_CHalf_inv@(Ct_C-n_sigma2*m*I)
	X = Ct_CHalf_inv@svd_r_proj(B, rank)
	
	return X

if __name__ == "__main__":
	
	synthetic = False
	
	if synthetic:
		m, n = 100, 100
		rank = 3
		mu = 5
		sigma2 = 1
		A = np.random.normal(mu, math.sqrt(sigma2), (m,n)) 	# generate gaussian matrix
		A_r = svd_r_proj(A, rank)							# rank-r truncation
		
		repeat = 50000
		X_diff = []
		A_diff = []
		for i in range(repeat):
			print(i)
			n_mu = 0
			n_sigma2 = 0.01**2
			G = np.random.normal(n_mu, math.sqrt(n_sigma2), (m,n))
			
			C=A+G

			X_star = closed_form(A, n_sigma2, rank)
			X_til = approximated_form(C, n_sigma2, rank)
#			print(LA.matrix_rank(X_star))
#			print(LA.matrix_rank(X_til))
#			print("|X_til-X_star|_F", LA.norm(X_til-X_star, 'fro'))
			X_diff.append(LA.norm(X_til-X_star, 2))

			A_hat = C@X_star
			A_hat_approx = C@X_til

#			print("|A_hat-A|_F with x_star", LA.norm(A_hat-A, 'fro'))
#			print("|A_hat_approx-A|_F with x_til", LA.norm(A_hat_approx-A, 'fro'))
#			print("|A_hat-A_hat_approx|_F", LA.norm(A_hat_approx-A_hat, 'fro'))
			A_diff.append(LA.norm(A_hat_approx-A_hat, 2))
		
		plt.hist(X_diff,bins='auto')
		plt.title("X_diff histogram")
		plt.xlabel('$||X^*-X_{tilde}||$')
		plt.ylabel('# of experiments')
		plt.show()
		plt.hist(A_diff,bins='auto')
		plt.title("A_diff histogram")
		plt.xlabel('$||CX^*-CX_{tilde}||$')
		plt.ylabel('# of experiments')
		plt.show()
	else:
		A = cv2.imread('rank40img.png', cv2.IMREAD_GRAYSCALE)
		Amax = np.max(A)
		Amin = np.min(A)
		Amean = np.mean(A)
		A_normalize = (A-Amean)/(Amax-Amin)
#		A_resize = cv2.resize(A,(100,100))
#		cv2.imwrite('knot.png',A_resize)
		m,n = A.shape
		rank = 40
		print(np.std(A))
#		U, s, Vt = LA.svd(A, full_matrices=False)
#		stable_r = np.around(np.sum(s**2)/(s[0]**2),decimals = 1)
#		grey_img = svd_r_proj(A, rank)
#		cv2.imwrite('knotrank5.png', grey_img)

		n_mu = 0
		n_sigma2 = 2**2
		n_sigma2_normalized = n_sigma2/((Amax-Amin)**2)
		G = np.random.normal(n_mu, math.sqrt(n_sigma2), (m,n))
		
		C=A+G
		cv2.imwrite('camerarank40noisy.png', C)
		Cmax = np.max(C)
		Cmin = np.min(C)
		Cmean = np.mean(C)
		C_normalize = (C-Cmean)/(Cmax-Cmin)
		n_sigma2_normalizedc = n_sigma2/((Cmax-Cmin)**2)
	#	cv2.imwrite('noisytrunc.png', svd_r_proj(A+G, 40))
		
		X_star = closed_form(A_normalize, n_sigma2_normalized, rank)
		X_til = approximated_form(C_normalize, n_sigma2_normalizedc, rank)
	#	print(LA.matrix_rank(X_star))
	#	print(LA.matrix_rank(X_til))
		print("|X_til-X_star|_F", LA.norm(X_til-X_star, 'fro'))
	#	print(A)
		A_hat_n = C_normalize@X_star
		A_hat_approx_n = C_normalize@X_til

		cv2.imwrite('recoveredbyAHAT.png', A_hat_n*(Amax-Amin)+Amean)
		cv2.imwrite('recovereedbyAPROX.png', A_hat_approx_n*(Amax-Amin)+Amean)

		print("|A_hat-A|_F with x_star", LA.norm(A_hat_n-A_normalize, 'fro'))
		print("|A_hat-A|_F with x_til", LA.norm(A_hat_approx_n-A_normalize, 'fro'))
	
	