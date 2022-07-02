from model import HyperResNet34
import torchvision.transforms as T
import numpy as np
import torch
from PIL import Image
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Test our model')
parser.add_argument('-m','--model', help='Path to the trained model', type=str ,required= True)
parser.add_argument('-i','--image', help='Path to the image', type=str, required= True)
parser.add_argument('-n','--nbClass', help='Number of class', type=int, required= True)
args = parser.parse_args()


def main(args):
	# model loading...

	model = HyperResNet34(args.nbClass)
	state = torch.load(args.model)
	model.load_state_dict(state['state_dict'])
	# optimizer.load_state_dict(state['optimizer'])
	model.eval()

	# Transformation 

	transform = T.Compose(
	   [
	    T.Resize((224, 224)),
	    T.ToTensor(),
	   ]
	)

	from torch.autograd.grad_mode import no_grad
	imag = Image.open(args.image).convert('RGB')
	imag = transform(imag)
	imag = np.expand_dims(imag, axis=0)
	imag = torch.from_numpy(imag)

	colors={
	    0: 'Black',
	    1: 'Blue',
	    2: 'Brown',
	    3: 'Green',
	    4: 'Orange',
	    5: 'Red',
	    6: 'Violet',
	    7: 'White',
	    8: 'Yellow'
	}



	with torch.no_grad():

	  out = model(imag)

	show = np.squeeze(imag.permute(0,2,3,1))

	plt.imshow(show)
	plt.axis('off')
	plt.title(colors[out.max(1)[1].cpu().item()])
	plt.show()

if __name__ == '__main__':

	main(args)