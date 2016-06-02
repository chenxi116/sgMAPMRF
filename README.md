# sgMAPMRF

This is a LP Relaxation-based MAP-MRF Solver using subgradient method. 

The main reference of the algorithm used in this code is:

	Nowozin, Sebastian, and Christoph H. Lampert. 
	"Structured learning and prediction in computer vision." 
	Foundations and Trends® in Computer Graphics and Vision 6.3-4 (2011): 185-365.

Right now the code includes an image denoising example. The input is a black-and-white image. The code first randomly flips each pixel to produce an image with noise. The solver returns a denoised image, which is compared against the original image to show the effectiveness.

See the `argparse` section of the code for meanings of parameters. In general, simply run:

	python sgMAPMRF.py -i convex.png