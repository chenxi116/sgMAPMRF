# Solving MAP-MRF Linear Programming Relaxation Using Subgradient Method
# Chenxi Liu
# May 2016
# Course project for UCLA EE236C

# Main reference: 
# Nowozin, Sebastian, and Christoph H. Lampert. 
# "Structured learning and prediction in computer vision." 
# Foundations and Trends in Computer Graphics and Vision 6.3-4 (2011): 185-365.

# Application: Image denoising
# Input: A black-and-white image whose path is specified with -i. 
#        See argparse part for meanings of other parameters.
# Output: A denoised image.

from scipy.io import loadmat
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import pdb
import matplotlib.pyplot as plt
import argparse
import random
import time


def sgMAPMRF(unary_e, pairwise_i, pairwise_e, m = 5., Tmax = 50, gap = 10.):

	nv = len(unary_e) # number of nodes (V)
	ne = len(pairwise_i) # number of edges (E)
	# in this example ne = 2 * nv

	# unary_e: numpy array of size nv * 2
	# pairwise_i: numpy array of ne * 2
	# pairwise_e: numpy array of size ne * 2 * 2

	# unary_mu: nv * 2 binary matrix. Each row has one 1, and the other is 0
	# pairwise_mu: ne * 2 * 2 binary tensor. Each 2*2 block has one 1
	# ld: ne * 2 * 2 tensor. The first 2 represents the order of node i j. 
	#     The second 2 represents state 0 and 1

	# Initialize unary_mu, pairwise_mu, ld
	unary_mu = np.concatenate((np.ones((nv, 1)), np.zeros((nv, 1))), axis = 1) # state 0
	pairwise_mu = np.array([np.array([[1, 1], [1, 1]])] * ne) # state 0
	ld = np.zeros((ne, 2, 2))
	
	unary_ld = GetUnaryLd(nv, pairwise_i)

	p, q = [], []
	t1 = time.time()
	for t in range(1, Tmax):

		# Update unary_mu
		t1 = time.time()
		unary_mu = UpdateUnary(unary_e, unary_ld, ld)
		t2 = time.time()
		print t2 - t1

		# Update pairwise_mu
		pairwise_mu = UpdatePairwise(pairwise_e, ld)

		cmtx = GetConstraintMtx(pairwise_i, unary_mu, pairwise_mu)

		# Compute objective
		dual_obj = DualObjective(unary_e, pairwise_e, unary_mu, pairwise_mu, ld, cmtx)
		q.append(dual_obj)
		
		primal_obj = PrimalObjective(unary_e, pairwise_i, pairwise_e, unary_mu)
		p.append(primal_obj)

		print '{:>9} {:>4} {:>6} {:10.2f} {:>8} {:10.2f}'.format \
				('Iteration', t, 'Dual', dual_obj, 'Primal', primal_obj)
		
		if primal_obj - dual_obj < gap:
			break

		# Update ld
		alpha = (1. + m)/(t + m) # step size
		ld = ld + alpha * cmtx # gradient ascent

	t2 = time.time()
	print 'Elapsed Time:', t2 - t1
	
	y = [np.argmax(item) for item in unary_mu]
	return y, p, q


def DualObjective(unary_e, pairwise_e, unary_mu, pairwise_mu, ld, cmtx):

	unary_term = np.multiply(unary_e, unary_mu).sum()
	pairwise_term = np.multiply(pairwise_e, pairwise_mu).sum()
	dual_term = np.multiply(ld, cmtx).sum()

	# print 'Unary', unary_term, 'Pairwise', pairwise_term, 'Dual', dual_term
	return unary_term + pairwise_term + dual_term


def PrimalObjective(unary_e, pairwise_i, pairwise_e, unary_mu):

	unary_term = np.multiply(unary_e, unary_mu).sum()

	ne = len(pairwise_i)
	pairwise_mu = np.zeros((ne, 2, 2))
	for idx in range(ne):
		i, j = pairwise_i[idx][0], pairwise_i[idx][1]
		s0 = np.argmax(unary_mu[i])
		s1 = np.argmax(unary_mu[j])
		pairwise_mu[idx, s0, s1] = 1
	pairwise_term = np.multiply(pairwise_e, pairwise_mu).sum()

	return unary_term + pairwise_term


def GetConstraintMtx(pairwise_i, unary_mu, pairwise_mu):

	ne = len(pairwise_i)
	cmtx = np.zeros((ne, 2, 2))

	i, j = pairwise_i[:, 0], pairwise_i[:, 1]
	cmtx[:, 0, 0] = pairwise_mu[:, 0, 0] + pairwise_mu[:, 0, 1] - unary_mu[i, 0]
	cmtx[:, 0, 1] = pairwise_mu[:, 1, 0] + pairwise_mu[:, 1, 1] - unary_mu[i, 1]
	cmtx[:, 1, 0] = pairwise_mu[:, 0, 0] + pairwise_mu[:, 1, 0] - unary_mu[j, 0]
	cmtx[:, 1, 1] = pairwise_mu[:, 0, 1] + pairwise_mu[:, 1, 1] - unary_mu[j, 1]

	return cmtx


def GetUnaryLd(nv, pairwise_i):

	unary_ld = []
	for iv in range(nv):
		unary_ld.append([])

	ne = len(pairwise_i)
	for ie in range(ne):
		i, j = pairwise_i[ie]
		unary_ld[i].append((ie, 0))
		unary_ld[j].append((ie, 1))
		
	return unary_ld


def UpdateUnary(unary_e, unary_ld, ld):

	nv = len(unary_e)
	unary_mu = np.zeros((nv, 2))
	for idx in range(nv):
		sum0, sum1 = 0, 0
		for item in unary_ld[idx]:
			sum0 += ld[item[0], item[1], 0]
			sum1 += ld[item[0], item[1], 1]
		s0 = unary_e[idx][0] - sum0
		s1 = unary_e[idx][1] - sum1
		if s0 < s1:
			unary_mu[idx][0] = 1
		else:
			unary_mu[idx][1] = 1

	return unary_mu


def UpdatePairwise(pairwise_e, ld):

	ne = len(pairwise_e)
	pairwise_mu = np.zeros((ne, 2, 2))
	mtx = np.zeros((ne, 2*2))
	mtx[:, 0] = pairwise_e[:, 0, 0] + ld[:, 0, 0] + ld[:, 1, 0]
	mtx[:, 1] = pairwise_e[:, 0, 1] + ld[:, 0, 0] + ld[:, 1, 1]
	mtx[:, 2] = pairwise_e[:, 1, 0] + ld[:, 0, 1] + ld[:, 1, 0]
	mtx[:, 3] = pairwise_e[:, 1, 1] + ld[:, 0, 1] + ld[:, 1, 1]
	idx = np.unravel_index(np.argmin(mtx, axis = 1), (2, 2))
	pairwise_mu[range(ne), idx[0], idx[1]] = 1

	return pairwise_mu


def VisualizeImage(image, title):

	plt.figure()
	plt.imshow(image, cmap='Greys')
	plt.title(title)
	plt.show()


def VisualizeConvergence(p, q):

	plt.figure()
	plt.plot(q, label='Dual Objective')
	plt.plot(p, label='Primal Objective')
	plt.legend()
	plt.xlabel('Iteration')
	plt.ylabel('Objective')
	plt.show()


def PrepFactors(noise, unary_w, pairwise_w):

	# Objective: argmin_y \sum_F E_F(y_F)
	# Unary:     E(y)        = 0          if y = x
	#                        = unary_w    if y \neq x
	# Pairwise:  E(y_1, y_2) = 0          if y_1 = y_2
	#                        = pairwise_w if y_1 \neq y_2

	# Image height and width
	[h, w] = np.shape(noise)

	# Unary energy
	unary_e = zip(noise.flatten(), 1 - noise.flatten())
	unary_e = np.multiply(unary_e, unary_w) 
	# unary_e[i][0]: unary energy of y_i = 0
	# unary_e[i][1]: unary energy of y_i = 1

	# Pairwise index
	pairwise_i = []
	for i in range(h * w):
		if i % w == w - 1:
			pair_right = (i, i + 1 - w)
		else:
			pair_right = (i, i + 1)
		pairwise_i.append(pair_right)
		if i + w >= h * w:
			pair_down = (i, i + w - h * w)
		else:
			pair_down = (i, i + w)
		pairwise_i.append(pair_down)
	pairwise_i = np.array(pairwise_i)

	# Pairwise energy
	pairwise_e = np.array([[0, 1], [1, 0]]) * pairwise_w
	pairwise_e = np.array([pairwise_e] * len(pairwise_i))

	return unary_e, pairwise_i, pairwise_e


def ConvertImage(path, width):

	image = imread(path)
	image_grey = rgb2gray(image)
	[h, w] = np.shape(image_grey)
	image_resize = resize(image_grey, (int(round(h * float(width) / w)), width))

	return np.uint8(image_resize < 0.5)


def AddNoise(image, n):

	[h, w] = np.shape(image)
	noise = np.copy(image)
	for i in range(h):
		for j in range(w):
			if random.random() < n:
				noise[i, j] = 1 - noise[i, j]

	return noise


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-m', type=float, default=5.) # step size (1 + m)/(t + m)
	parser.add_argument('-t', type=int, default=50) # maximum iteration
	parser.add_argument('-g', type=float, default=10.) # primal-dual gap threshold
	parser.add_argument('-u', type=float, default=1.) # unary_w
	parser.add_argument('-p', type=float, default=1.) # pairwise_w
	parser.add_argument('-i', type=str, default='convex.png') # original image
	parser.add_argument('-w', type=int, default=320) # image width after resize
	parser.add_argument('-n', type=float, default=.1) # noise percentage
	parser.add_argument('-v', type=bool, default=False) # whether to visualize
	args = parser.parse_args()

	image = ConvertImage(args.i, args.w) # Convert image to binary
	noise = AddNoise(image, args.n)
	unary_e, pairwise_i, pairwise_e = PrepFactors(noise, args.u, args.p)

	imsave('image.png', (1 - image) * 255)
	imsave('noise.png', (1 - noise) * 255)
	if args.v:
		# Visualize ground truth image and image with noise
		VisualizeImage(image, 'Original Image')
		VisualizeImage(noise, 'Noise Image')

	# Call subgradient method MAP-MRF solver
	y, p, q = sgMAPMRF(unary_e, pairwise_i, pairwise_e, args.m, args.t, args.g)
	denoise = np.reshape(y, np.shape(noise))

	imsave('denoise.png', (1 - denoise) * 255)
	if args.v:
		# Visualize convergence and denoise image
		VisualizeConvergence(p, q)
		VisualizeImage(denoise, 'Denoise Image')

	# Print number of pixels different from original image
	print 'Pixel Error:', abs(np.subtract(y, image.flatten())).sum()

