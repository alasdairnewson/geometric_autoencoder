
import tensorflow as tf
import numpy as np
import math
import pdb
import scipy.misc
import scipy
import sys
import glob, os
import pickle
import random
import imageio
from skimage import color


from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import rc

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose
from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
from keras import applications
from keras.layers import Input
from keras import regularizers
from keras import initializers

def read_image(file_name):
	img_out = imageio.imread(file_name)
	return img_out

def write_image(img,file_name):
	# clamp values
	img = np.maximum(np.minimum(255.0*img,255.0),0.0)
	imageio.imwrite(file_name, np.uint8(img))

def display_parameters(parameter_dict):
	
	print('')
	for key, value in parameter_dict.items():
		print(key, ' : ', value)


def normalise_image(img_in):
	img_out=img_in
	for i in range(0,img_in.shape[2]):
		img_out[:,:,i] = img_out[:,:,i] - img_out[:,:,i].min()
		img_out[:,:,i] = img_out[:,:,i] / img_out[:,:,i].max()

	return img_out


def retrieve_dataset(dataset_dir):
	file_list = sorted(glob.glob(dataset_dir+"*.png"))
	m,n,_ = read_image(file_list[0]).shape #scipy.misc.imread(file_list[0],flatten=True,mode='RGB').shape
	c = 1

	train_data = np.zeros((len(file_list),m,n,c))
	for i in range(0,len(file_list)):
		# note, rgb2gray converts to the interval (0.0,1.0)
		img_temp = color.rgb2gray(read_image(file_list[i]))#scipy.misc.imread(file_list[i],flatten=True,mode='RGB')/255.0
		train_data[i,:,:,0] = img_temp[:,:]
	return train_data

def write_manifold(ae_model,dataset_test,file_name):

	output_ae = ae_model.predict(dataset_test)
	input_ae = ae_model.input

	area = np.power( dataset_test.sum(axis=(1,2,3)) ,1.0) #np.power(r,4.0) #
	z_array = np.zeros((area.shape[0],1))

	# get output of code (z) layer
	middle_layer = int(np.floor(len(ae_model.layers)/2.0))
	z_layer = ae_model.layers[middle_layer].output
	#create a function to extract this layer
	z_layer_fun = K.function([input_ae], [z_layer])

	z_array = z_layer_fun([dataset_test])[0].squeeze()

	# z_array = z_array[50:200]
	# area = area[50:200]

	inds_sort = np.argsort(z_array)
	z_array = z_array[inds_sort]
	area = area[inds_sort]

	plt.figure()
	axis_font_size = 20
	plt.plot(z_array,area,linewidth=4)
	plt.xlabel("z",fontsize=axis_font_size)
	plt.ylabel("area", rotation=90, labelpad=15,fontsize=axis_font_size)
	plt.tight_layout()
	plt.savefig(file_name)
	plt.close()

	return

def write_encoder_weights(sess,ae,img_in,img_name):
	encoder = sess.run(ae['encoder'], feed_dict={ae['x']: img_in})
	#
	n_layers = len(encoder)
	for i_layer in range(0,n_layers):
		curr_weights = encoder[i_layer]
		write_weight_image(curr_weights,img_name+'_autoencoding_layer_'+str(i_layer))

def write_weight_image(img_in,img_name):
	from mpl_toolkits.mplot3d.axes3d import Axes3D

	output_image_dim = 3
	if (output_image_dim ==2):
		img_filter_resize = 64
		img_space = 2

		img_out = np.zeros(( (img_filter_resize+img_space)*img_in.shape[3],(img_filter_resize+img_space)*img_in.shape[2]))
		i_index = 0
		j_index = 0
		for i_response in range(0,img_in.shape[3]):
			j_index = 0
			for j_response in range(0,img_in.shape[2]):
				curr_img = np.squeeze(img_in[:,:,j_response,i_response])
				curr_img = curr_img-curr_img.min()
				img_out[ i_index:(i_index+img_filter_resize), j_index:(j_index+img_filter_resize)] = \
					scipy.misc.imresize(curr_img,(img_filter_resize,img_filter_resize),'nearest')
				j_index = j_index+img_filter_resize+img_space
			i_index = i_index+img_filter_resize+img_space

		#create colour image
		#pdb.set_trace()
		img_out = np.tile(img_out[:,:,None], [1,1,3])
		maxVal = img_out.max()

		i_index = img_filter_resize
		j_index = 0
		for i_response in range(0,img_in.shape[3]):
			j_index = img_filter_resize
			for j_response in range(0,img_in.shape[2]):
				img_out[ :, j_index:(j_index+img_space),1] = maxVal
				j_index = j_index+img_filter_resize+img_space
			img_out[ i_index:(i_index+img_space), :,1] = maxVal
			i_index = i_index+img_filter_resize+img_space

		#add first border
		img_out_temp = np.zeros((img_out.shape[0]+img_space,img_out.shape[1]+img_space,3))
		img_out_temp[img_space:,img_space:,:] = img_out
		img_out_temp[0:img_space,:,1] = maxVal
		img_out_temp[:,0:img_space,1] = maxVal

		scipy.misc.imsave(img_name+'_encoder_weights.png',img_out_temp) #s
	else:
		m = img_in.shape[3]
		n = img_in.shape[2]

		x = np.arange(0,3)
		y = np.arange(0,3)
		xs,ys = np.meshgrid(x,y)
		ax = np.zeros((m*n,1))

		filter_size = 3
		z_lim = 1
		zs = np.zeros((filter_size*filter_size,1))

		fig = plt.figure(figsize=plt.figaspect(m/n))

		for i in range(0,m):  #output size (number of filters)
			for j in range(0,n):  #input size (depth of the filters)
				#ax[i*n+j] = fig.add_subplot(i+1,j+1,1, projection='3d')
				ax = fig.add_subplot(m,n,i*n+j+1, projection='3d')
				curr_img = (img_in[:,:,j,i]).flatten()
				ax.bar3d(xs.flatten(), ys.flatten(), zs.flatten(), np.squeeze(np.ones((filter_size*filter_size,1))),\
					np.squeeze(np.ones((filter_size*filter_size,1))), curr_img.flatten(), color='#00ceaa')
				ax.set_zlim3d(-z_lim, z_lim)
	
		plt.savefig(img_name+'_encoder_weights.png')
	plt.close()



def plot_interpolation(sess,ae,data,first_image_id,second_image_id,file_list,graph_output_name=""):

	y,z = sess.run([ae['y'],ae['z']], feed_dict={ae['x']: np.asarray(data)})
	z = np.squeeze(z)
	y = np.squeeze(y)

	if (len(z.shape) == 1):
		z_size = 1
		z = np.expand_dims(z,axis=1)
		data_temp = np.hstack((z,z))
	else:
		z_size = z.shape[1]

	img_size = 64
	num_examples = 100
	decimalPlaces = 3
	currInd = 0
	# first z
	z_first = z[first_image_id,:]
	#second z
	z_second = z[second_image_id,:]

	z_list = z_first + np.expand_dims(np.linspace(0,1,num_examples),axis=1)*(z_second-z_first)
	x_list_first = sess.run(ae['y'], feed_dict={ae['z']: np.reshape(z_list,(z_list.shape[0],1,1,z_size))})

	n_iters = 10
	z_list_iter = z_list
	x_list = x_list_first
	for ind_i in range(0,n_iters):
		if (z_size == 1):
			x_list_temp = sess.run(ae['y'], feed_dict={ae['z']: np.reshape(z_list_iter[:,0],(z_list_iter.shape[0],1,1,z_size))})
		else:
			x_list_temp = sess.run(ae['y'], feed_dict={ae['z']: np.reshape(z_list_iter,(z_list_iter.shape[0],1,1,z_size))})
		x_list = np.zeros((x_list_temp.shape[0],img_size*img_size))
		for ind_i in range(0,x_list.shape[0]):
			x_list[ind_i,:] = np.reshape(x_list_temp[ind_i,:,:],(1,img_size*img_size))
		for ind_i in range(0,x_list.shape[0]):
			x_list[ind_i,:] = x_list[ind_i,:]-np.min(x_list[ind_i,:].flatten())
			x_list[ind_i,:] = x_list[ind_i,:]/(np.max(x_list[ind_i,:].flatten()))
		z_list_iter = sess.run(ae['z'], feed_dict={ae['x']: np.asarray(x_list)})

	if (len(z_list_iter.shape) == 1):
		z_list_iter = np.expand_dims(z_list_iter,axis=1)
	else:
		z_list_iter = np.squeeze(z_list_iter)
	# plot result
	fig = plt.figure()
	ax = fig.add_subplot(111)

	#set up colours
	first_colour = 1
	second_colour = 5
	third_colour = 10
	colour_list = np.vstack( (first_colour*np.ones((z.shape[0],1)) , \
		second_colour*np.ones((z_list.shape[0],1)),  third_colour*np.ones((z_list_iter.shape[0],1))))

	plot_data = np.vstack( (z[:,0:z_size],z_list[:,0:z_size],z_list_iter[:,0:z_size]) )
	img_data = np.vstack( (y,np.reshape(x_list_first,(x_list.shape[0],img_size,img_size)),np.reshape(x_list,(x_list.shape[0],img_size,img_size))))
	if (z_size == 1):
		plot_data = np.hstack((plot_data,plot_data))
		z = np.hstack((z,z))

	line = ax.scatter(plot_data[:, 0], plot_data[:, 1], s=10,c=colour_list)#,picker=True)
	if (graph_output_name==""):
		# create the annotations box
		im = OffsetImage(img_data[0,:,:], zoom=5)
		xybox=(50., 50.)
		ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
				boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
		# add it to the axes and make it invisible
		ax.add_artist(ab)
		ab.set_visible(False)

		def on_pick(event):
			# if the mouse is over the scatter points

			# find out the index within the array from the event
			#ind = line.contains(event)[1]["ind"]
			xdata, ydata = line.get_data()

			# get the figure size
			w,h = fig.get_size_inches()*fig.dpi
			ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
			hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
			# if event occurs in the top or right quadrant of the figure,
			# change the annotation box position relative to mouse.
			ab.xybox = (xybox[0]*ws, xybox[1]*hs)
			# make annotation box visible
			ab.set_visible(True)
			# place it at the position of the hovered scatter point
			ab.xy =(plot_data[ind[0],0], plot_data[ind[0],1])
			# set the image corresponding to that point
			im.set_data(img_data[ind[0],:,:])

		# 	fig.canvas.draw_idle()
		def hover(event):
			# if the mouse is over the scatter points
			if line.contains(event)[0]:
				# find out the index within the array from the event
				ind = line.contains(event)[1]["ind"]
				# get the figure size
				w,h = fig.get_size_inches()*fig.dpi
				ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
				hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
				# if event occurs in the top or right quadrant of the figure,
				# change the annotation box position relative to mouse.
				ab.xybox = (xybox[0]*ws, xybox[1]*hs)
				# make annotation box visible
				ab.set_visible(True)
				# place it at the position of the hovered scatter point
				ab.xy =(plot_data[ind[0],0], plot_data[ind[0],1])
				# set the image corresponding to that point
				im.set_data(img_data[ind[0],:,:])
			else:
				#if the mouse is not over a scatter point
				ab.set_visible(False)
			fig.canvas.draw_idle()


		# add callback for mouse moves
		fig.canvas.mpl_connect('motion_notify_event', hover)
		#fig.canvas.mpl_connect('pick_event', on_pick)
		plt.show()
	else:
		plt.savefig(graph_output_name)


def create_directory(directory):
	if (os.path.isdir(directory)==0):
		os.mkdir(directory)

def create_param_id_file_and_dir(param_dir):
	import time
	ts = time.time()
	#find if id already exists
	param_name = str(ts)
	os.mkdir(param_dir+param_name)

	return param_name,param_dir+param_name

#go through the 
def find_model_with_id(model_id,model_dir):
	for x in os.listdir(model_dir):
		curr_dir = model_dir+x
		curr_shape = x
		for y in os.listdir(curr_dir):
			if (y == model_id):	#
				return curr_shape
	print("Error, model id not found")
	return ""


def parse_pnorm_filename(fileName):
	rIndexBegin = fileName.index('_r_')
	rIndexEnd = fileName.index('_p_')
	r = int(fileName[(rIndexBegin+3):(rIndexEnd)])

	pIndexBegin = fileName.index('p_')
	pIndexEnd = fileName.index('.png')
	p = int(fileName[(pIndexBegin+3):(pIndexEnd)])
	return r,p

def parse_file_name(fileName):
	base=os.path.basename(fileName)
	out_name = os.path.splitext(base)[0]

	return out_name

def parse_disk_square_filename(fileName):
	rIndexBegin = fileName.index('_radius')
	rIndexEnd = fileName.index('.png')
	r = int(fileName[(rIndexBegin+8):(rIndexEnd)])

	return r

def parse_disk_radius(fileName):
	n_leading_zeros = 6
	begin_string = '_number_'
	rIndexBegin = fileName.index(begin_string)
	r = int(fileName[(rIndexBegin+len(begin_string)):(rIndexBegin+len(begin_string)+n_leading_zeros)])

	return r


def shuffle_list(list_in,random_seed):
	list_shuffle = list_in.copy()
	random.Random(random_seed).shuffle(list_shuffle)
	return list_shuffle

def get_parameters(param_file_name):

	parameter_struct = {}
	with open(param_file_name, 'rb') as handle:
		parameter_dict = pickle.load(handle)

		for key, value in parameter_dict.items():
			parameter_struct[key] = value

		return parameter_struct