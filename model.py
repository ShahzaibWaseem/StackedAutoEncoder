import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 10

def loss_func(pred, target):
	return F.binary_cross_entropy(pred, target)

class Autoencoder(nn.Module):
	""" Denoising Autoencoder """
	def __init__(self, input_size, output_size, depth, channel_downsample, spatial_neurons, individual_training=False, debug=False):
		super(Autoencoder, self).__init__()
		hidden_size = input_size // (channel_downsample * depth)

		self.encoder = self.autoencoder_layers(input_size, stride=1, bottleneck_factor=channel_downsample, depth=depth, encoder=True)
		self.decoder = self.autoencoder_layers(hidden_size, stride=1, bottleneck_factor=2, depth=depth, encoder=False)

		self.individual_training = individual_training
		self.debug = debug

		if self.individual_training:
			self.criterion = nn.CrossEntropyLoss()
			self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

		# Channel Neuron Calculation
		self.n_neurons = hidden_size * spatial_neurons
		print("Linear Layer\t%i\t%i" % (self.n_neurons, N_CLASSES))

		self.encoder_classifier = nn.Linear(self.n_neurons, N_CLASSES)

	def autoencoder_layers(self, input_size, stride, bottleneck_factor, depth, encoder):
		if depth < 0:
			depth = 0

		self.hidden_size = input_size

		layers = []
		if encoder:
			for _ in range(depth):
				print("encoder\t%i\t%i\t%i" % (_, self.hidden_size, self.hidden_size//bottleneck_factor))
				layers.append(nn.Conv2d(self.hidden_size, self.hidden_size//bottleneck_factor, kernel_size=3, stride=stride, padding=1))
				layers.append(nn.ReLU())
				self.hidden_size = self.hidden_size//bottleneck_factor
		else:
			for _ in range(depth):
				print("decoder\t%i\t%i\t%i" % (_, self.hidden_size, self.hidden_size*bottleneck_factor))
				layers.append(nn.ConvTranspose2d(self.hidden_size, self.hidden_size*bottleneck_factor, kernel_size=3, stride=1, padding=1))
				layers.append(nn.ReLU())
				self.hidden_size = self.hidden_size*bottleneck_factor

		return nn.Sequential(*layers)

	def forward(self, x, label):
		x = x.detach()
		bottleneck = self.encoder(x)

		if self.debug:
			print("x shape:", x.shape, "\tbottleneck shape:", bottleneck.shape, end="\t")

		y = self.decoder(bottleneck)
		y = y.detach()

		bottleneck = bottleneck.view(bottleneck.size(0), -1).detach()

		if self.debug:
			print("bottleneck flatten shape:", bottleneck.shape)

		classification = self.encoder_classifier(bottleneck)

		if self.individual_training:
			self.optimizer.zero_grad()
			loss = self.criterion(classification, label)
			loss.backward()
			self.optimizer.step()
		
		return y, bottleneck, classification

class StackedAutoencoder(nn.Module):
	IMAGE_SIZE = 32
	spatial_neurons = IMAGE_SIZE

	def __init__(self, input_size, output_size, depths = [2, 2, 2], noise=False, channel_downsample=2, spatial_downsample=1, verbose_printing=True):
		super(StackedAutoencoder, self).__init__()
		self.channel_downsample = channel_downsample
		self.spatial_downsample = spatial_downsample
		self.verbose_printing = verbose_printing
		self.noise = noise

		self.input_conv = nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1)
		self.input_relu = nn.ReLU()

		print("\nAuto Encoder 1", end="\n\n")
		self.autoencoder1, self.downsample_module1 = self.create_autoencoder(output_size, output_size, depths[0], self.channel_downsample)
		print("\nAuto Encoder 2", end="\n\n")
		self.autoencoder2, self.downsample_module2 = self.create_autoencoder(output_size*2, output_size*2, depths[1], self.channel_downsample)
		print("\nAuto Encoder 3", end="\n\n")
		self.autoencoder3, self.downsample_module3 = self.create_autoencoder(output_size*4, output_size*4, depths[2], self.channel_downsample)

		n_neurons = self.autoencoder1.n_neurons + self.autoencoder2.n_neurons + self.autoencoder3.n_neurons

		self.overall_classification = nn.Linear(n_neurons, N_CLASSES)

	def create_autoencoder(self, input_size, output_size, depth, channel_downsample):
		# Spatial Neuron Calculation
		self.spatial_neurons = self.spatial_neurons // (self.spatial_downsample * depth) if self.spatial_downsample > 1 else self.IMAGE_SIZE
		# print("Spatial Neurons", self.spatial_neurons)
		n_neurons = self.spatial_neurons * self.spatial_neurons		# height * width

		downsample_stream = []
		if channel_downsample > 1:
			for _ in range(depth-1):
				downsample_stream.append(nn.Conv2d(input_size, input_size, kernel_size=3, stride=self.spatial_downsample, padding=1))
				downsample_stream.append(nn.ReLU())
			downsample_stream.append(nn.Conv2d(input_size, input_size, kernel_size=3, stride=self.spatial_downsample, padding=1))

		downsample_module = nn.Sequential(*downsample_stream)
		return Autoencoder(input_size, output_size, depth, channel_downsample, spatial_neurons=n_neurons, debug=self.verbose_printing), downsample_module
	
	def add_noise(self, input):
		return input * (torch.autograd.Variable(input.data.new(input.size()).normal_(0, 0.1)) > -0.1).type_as(input)
	
	def forward(self, x, label):
		x = self.input_conv(x)
		# x = self.input_relu(x)

		x_copy = x

		# add noise to the input
		if self.noise:
			x = self.add_noise(x)

		out_ae1, bottleneck1, y1 = self.autoencoder1(x, label)

		if self.channel_downsample > 1:
			x = self.downsample_module1(x_copy)

		in_ae2 = torch.cat((out_ae1, x), 1)
		in_ae2_copy = in_ae2

		if self.noise:
			in_ae2 = self.add_noise(in_ae2)

		out_ae2, bottleneck2, y2 = self.autoencoder2(in_ae2, label)

		if self.channel_downsample > 1:
			in_ae2 = self.downsample_module2(in_ae2_copy)

		in_ae3 = torch.cat((out_ae2, in_ae2), 1)

		if self.noise:
			in_ae3 = self.add_noise(in_ae3)

		out_ae3, bottleneck3, y3 = self.autoencoder3(in_ae3, label)

		bottleneck = torch.cat((bottleneck1, bottleneck2, bottleneck3), 1)
		y = self.overall_classification(bottleneck)

		if self.verbose_printing:
			y = (y, y1, y2, y3)
		return y