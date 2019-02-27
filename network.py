import torch
import torch.nn as nn

class Generator(nn.Module):

	def __init__(self):
		super(Generator, self).__init__()

		# channel number in the feature layer
		f_ch = 32

	    # g function parameter
	    c = 16
	    w = 16
	    h = 16
	    g1_conv = input_len - 1 - h
	    g2_conv = (int(input_len / 2) - 1) - 1 - h
	    g3_conv = int((int(input_len / 2) - 1) / 2) - 1 - 1 - h
	    g4_deconv = h - ( int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1 - 1 - 2)
	    g5_deconv = h - ( int((int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1 - 1 - 2)
	    g6_deconv = h - ( int((int((int((int((int(input_len / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1) / 2) - 1 - 1)

	    self.down_sampler=nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(4, 4),stride=(2, 2),padding=(1,1))


	    #ENCODER

		self.en_conv_1=nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=f_ch, kernel_size=(3, 3)),  # f function   
			#32x126x126
			nn.LeakyReLU(0.2),
			nn.Conv2d(in_channels=f_ch, out_channels=c, kernel_size=(5,5), stride=(4,4)),
			nn.Conv2d(in_channels=c, out_channels=c, kernel_size=(3,3), stride=(2,2),padding=(1,1))
			#16x16x16
		)

		self.en_conv_2=nn.Sequential(
			#3x64x64
			nn.Conv2d(in_channels=3, out_channels=f_ch, kernel_size=(3, 3)),  # f function
			#32x62x62
			nn.LeakyReLU(0.2),
			nn.Conv2d(in_channels=f_ch, out_channels=c, kernel_size=(5,5),stride=(4,4),padding=(2,2)) # g function
			#16x16x16
		)

		self.en_conv_3=nn.Sequential(
			#3x32x32
			nn.Conv2d(in_channels=3, out_channels=f_ch, kernel_size=(3, 3),padding=(1,1)),  # f function
			#32x32x32
			nn.LeakyReLU(0.2),
			nn.Conv2d(in_channels=f_ch, out_channels=c, kernel_size=(3, 3), stride=(2,2),padding=(1,1)) # g function
		)


		self.en_conv_4=nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=f_ch, kernel_size=(3, 3),padding=(1,1)),  # f function
			nn.LeakyReLU(0.2),
			nn.Conv2d(in_channels=f_ch, out_channels=c, kernel_size=(3,3),padding=(1,1)) # g function
		)

		self.en_conv_5=nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=f_ch, kernel_size=(3, 3),padding=(1,1)),  # f function
			#32x8x8
			nn.LeakyReLU(0.2),
			nn.ConvTranspose2d(in_channels=f_ch, out_channels=c, kernel_size=(4,4),stride=(2,2),padding=(1,1)) # g function
			#16x16x16
		)

		self.en_conv_6=nn.Sequential(
			#3x4x4
			nn.Conv2d(in_channels=3, out_channels=f_ch, kernel_size=(3, 3),padding=(1,1)),  # f function
			#32x4x4
			nn.LeakyReLU(0.2),
			nn.ConvTranspose2d(in_channels=f_ch, out_channels=c,kernel_size=(4,4),stride=(4,4)) # g function
			#16x16x16
		)
		
		#16x16x16
		self.en_g_final=nn.Conv2d(in_channels=c, out_channels=c, kernel_size=(3,3))
		#16x14x14

		#DECODER
		#16x14x14
		self.de_g=nn.ConvTranspose2d(in_channels=c, out_channels=c, kernel_size=(3, 3))
		#16x16x16

		self.de_conv_6=nn.Sequential(
			#g function 
			nn.Conv2d(in_channels=c, out_channels=c, kernel_size=(1,1)),
			nn.ConvTranspose2d(in_channels=c, out_channels=f_ch, kernel_size=(1, 1)),
			nn.LeakyReLU(0.2),
			nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(4, 4), stride=(2, 2))
		)


	def forward(self, x):
		x1=x
		g1=self.en_conv_1(x1)
		x2=self.down_sampler(x1)
		g2=self.en_conv_2(x2)
		x3=self.down_sampler(x2)
		g3=self.en_conv_3(x3)
		x4=self.down_sampler(x3)
		g4=self.en_conv_4(x4)
		x5=self.down_sampler(x4)
		g5=self.en_conv_5(x5)
		x6=self.down_sampler(x5)
		g6=self.en_conv_6(x6)
		encoded=g1+g2+g3+g4+g5+g6
		encoded=self.en_g_final(encoded)


	def encoder(self):



	def decoder():
		pass