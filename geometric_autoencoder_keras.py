

import numpy as np
import scipy.misc
import pdb
import glob
from image_utils import *


result_dir = 'results/'
model_dir = 'models/'

def gradient_z(ae_model,parameter_dict,dataset_dir):

	# get output of code (z) layer
	img_backend = ae_model.input
	middle_layer = int(np.floor(len(ae_model.layers)/2.0))
	z_layer = ae_model.layers[middle_layer].output[:,0,0,0]
	z_grad = K.gradients(z_layer,img_backend)[0]
	get_grads = K.function([img_backend],[z_grad])

	#get dataset
	dataset = retrieve_dataset(dataset_dir)

	epsilon = 0.0001

	for i in range(0,dataset.shape[0]):
		img_in = np.expand_dims(dataset[i,:,:],axis=0)
		img_out = get_grads([img_in])[0].squeeze()**2

		img_out = img_out-img_out.min()
		img_out = img_out/(img_out.max())

		write_image(img_in.squeeze(),result_dir+parameter_dict['geometric_object']+'/'+parameter_dict['model_id']+'/'+'img_grad_z_'+str(i).zfill(4)+'_in.png')
		write_image(img_out.squeeze(),result_dir+parameter_dict['geometric_object']+'/'+parameter_dict['model_id']+'/'+'img_grad_z_'+str(i).zfill(4)+'_out.png')


def save_model_weights(model,model_file):
	model.save_weights(model_file)

def load_model_weights(model,model_file):
	model.load_weights(model_file)

def create_ae_model(input_shape,parameter_dict, mode_id=''):
	ae_model = Sequential()
	regularisation = regularizers.l2(parameter_dict['lambda_regularisation'])

	strides_shape = (2,2)
	kernel_shape = (3,3)

	input_ae = Input(shape=parameter_dict['input_shape'])
	z = input_ae

	# encoder
	for i in range(1,len(parameter_dict['filter_sizes'])):
		if(regularisation_type == 1 or regularisation_type == 2):
			z = Conv2D(parameter_dict['filter_sizes'][i], kernel_size=kernel_shape, strides=strides_shape,\
				kernel_regularizer=regularisation, activation=None, padding='valid')(z) 
		else:
			z = Conv2D(filters=parameter_dict['filter_sizes'][i], kernel_size=kernel_shape, strides=strides_shape,\
				activation=None, padding='valid')(z)
		#z = Activation('relu')(z)
		z = LeakyReLU(alpha=parameter_dict['alpha_lrelu'])(z)#LeakyReLU(alpha=0.0)(z)#

	#gradient of z wrt input image
	
	z_grad = K.gradients(z,input_ae)[0]
	y = z
	# decoder
	for i in np.arange(len(parameter_dict['filter_sizes'])-2,-1,-1):
		if(regularisation_type == 1):
			y = Conv2DTranspose(parameter_dict['filter_sizes'][i], kernel_size=kernel_shape, strides=strides_shape,\
				kernel_regularizer=regularisation, activation=None, padding='valid')(y)
		else:
			y = Conv2DTranspose(filters=parameter_dict['filter_sizes'][i],\
				kernel_size=kernel_shape, strides=strides_shape, activation=None, padding='valid')(y)
		#y = Activation('relu')(y)
		y = LeakyReLU(alpha=parameter_dict['alpha_lrelu'])(y)#
	
	output_ae = y
	ae_model = Model(input_ae,output_ae)

	optimizer = Adam(parameter_dict['learning_rate']) #

	if (model_id == ''): 	#create a new model
		# mse loss
		mse_loss = K.mean(K.square(input_ae - output_ae), axis=(0,1,2,3))

		if (regularisation_type < 3): #no regularisation, weight regularisation
			ae_loss = mse_loss
		elif(regularisation_type == 3):
			# z gradient loss
			z_grad_loss = K.mean(K.square(z_grad), axis=(0,1,2,3))
			ae_loss = mse_loss + parameter_dict['lambda_regularisation']*z_grad_loss

		ae_model.add_loss(ae_loss)
		ae_model.compile(optimizer=optimizer)
	else:	#load a previous one
		ae_model.load_weights(model_dir+parameter_dict['geometric_object']+'/'+parameter_dict['model_id']+'/'+parameter_dict['model_id']+'_weights.h5')
		ae_model.compile(optimizer=optimizer,loss='mse')
	ae_model.summary()

	return ae_model

def write_ae_output(ae_model,parameter_dict,dataset):

	#get dataset
	imgs_out = ae_model.predict([dataset])

	sigma = 0.0

	for i in range(0,dataset.shape[0]):
		img_in = dataset[i,:,:,:].squeeze()#+sigma*np.random.normal(size=dataset[i,:,:,:].squeeze().shape)
		img_out = imgs_out[i,:,:,:].squeeze()
		
		img_out = np.maximum(np.minimum(img_out,1.0),0.0)

		write_image(img_in[:,:],result_dir+parameter_dict['geometric_object']+'/'+parameter_dict['model_id']+'/'+'img_'+str(i).zfill(4)+'_in'+'.png')
		write_image(img_out[:,:],result_dir+parameter_dict['geometric_object']+'/'+parameter_dict['model_id']+'/'+'img_'+str(i).zfill(4)+'_out'+'.png')

def train_ae(dataset_train_dir, dataset_test_dir, model_id='',\
	geometric_object=None,regularisation_type=0,lambda_regularisation=1.0,epochs=10000):

	# parameters

	# default hyperparameters
	filter_sizes = (1, 8, 4, 3, 2, 1)
	strides_shape = (2,2)
	kernel_shape = (3,3)
	alpha = 0.05
	learning_rate = 0.01
	batch_size = 64

	# get the necessary datasets
	dataset_train = retrieve_dataset(dataset_train_dir)
	dataset_test = retrieve_dataset(dataset_test_dir)
	data_inds = np.asarray(range(0,dataset_train.shape[0]))

	input_shape = dataset_train.shape[1:4]

	#if necessary, create a new model_id
	#create the directories concerning the training object if necessary
	if (os.path.isdir(model_dir+geometric_object+"/")==0):
		os.mkdir(model_dir+geometric_object)
		os.mkdir(result_dir+geometric_object)
	if (model_id == ''):
		model_id, param_dir = create_param_id_file_and_dir(model_dir+geometric_object+'/')
		# create result dir
		os.mkdir(result_dir+geometric_object+'/'+model_id+'/')
		parameter_dict = {'model_id':model_id,'input_shape':input_shape, 'filter_sizes':filter_sizes,
			'learning_rate':learning_rate,'geometric_object':geometric_object,'alpha_lrelu':alpha,
			'epochs':epochs, 'lambda_regularisation':lambda_regularisation, 'regularisation_type': regularisation_type}
		with open(model_dir+'/'+parameter_dict['geometric_object']+'/'+parameter_dict['model_id']+'/'+parameter_dict['model_id']+'.pickle', 'wb') as handle:
			pickle.dump(parameter_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
		#now actually create the model
		ae_model = create_ae_model(dataset_train.shape[1:4],parameter_dict)
	else:
		parameter_dict = get_parameters(model_dir+parameter_dict['geometric_object']+'/'+parameter_dict['model_id']+'/'+parameter_dict['model_id']+'.pickle')
		#now actually create the model
		ae_model = create_ae_model(dataset_train.shape[1:4],parameter_dict,model_id)


	best_loss = float('inf')
	step_size = 500
	display_parameters(parameter_dict)

	for i in range(0,parameter_dict['epochs']):
		np.random.shuffle(data_inds)
		curr_batch = dataset_train[data_inds[0:batch_size],:,:,:]

		curr_loss = ae_model.train_on_batch(curr_batch,None)
		print("i : ",i, 'loss : ', curr_loss)
		if (i%step_size==0):
			write_manifold(ae_model,dataset_test,result_dir+parameter_dict['geometric_object']+\
				'/'+parameter_dict['model_id']+'/'+'iteration_'+str(i).zfill(6)+'.png')
			test_results_ae(ae_model,parameter_dict,dataset_test_dir)
			if (curr_loss<best_loss):
				best_loss = curr_loss
				save_model_weights(ae_model,\
					'models/'+parameter_dict['geometric_object']+'/'+parameter_dict['model_id']+'/'+parameter_dict['model_id']+'_weights.h5')
	

def test_results_ae(ae_model,parameter_dict,dataset_dir):

	# test
	dataset_test = retrieve_dataset(dataset_dir)
	input_shape = dataset_test.shape[1:4]

	write_manifold(ae_model,dataset_test,result_dir+parameter_dict['geometric_object']+\
				'/'+parameter_dict['model_id']+'/'+'manifold.png')
	
	#gradient_z(ae_model,parameter_dict,dataset_dir)

	write_ae_output(ae_model,parameter_dict,dataset_test)

def test_ae(dataset_dir,model_id):

	parameter_struct = get_parameters(model_dir+geometric_object+'/'+model_id+'/'+model_id+'.pickle')
	display_parameters(parameter_struct)

	# create result directory if necessary
	if (os.path.isdir(result_dir+geometric_object+'/')==0):
		os.mkdir(result_dir+geometric_object+'/')

	curr_result_dir = result_dir+geometric_object+'/'+model_id+'/'
	if (os.path.isdir(curr_result_dir)==0):
		os.mkdir(curr_result_dir)

	ae_model = create_ae_model(parameter_struct['input_shape'],parameter_struct,model_id)
	test_results_ae(ae_model,parameter_struct,dataset_dir)


if __name__ == '__main__':
	
	flags = tf.app.flags
	flags.DEFINE_integer("is_training", 1, "Training or testing [True]")
	flags.DEFINE_string("model_id", "", "ID for identifying the parameter set")
	flags.DEFINE_string("geometric_object", "disk", "The type of geometric training object")
	flags.DEFINE_integer("regularisation_type", 0, "The type of regularisation : 0 (none), 1 (encoder and decoder weights), 2 (encoder weights), 3 (contractive)")
	flags.DEFINE_float("lambda_reg",1.0,"Regularisation strength parameter")
	flags.DEFINE_integer("epochs",10000,"Epochs")
	flags.DEFINE_boolean("use_gpu","1","Whether we should use the gpu (1)")

	FLAGS = flags.FLAGS

	if (FLAGS.use_gpu == 0):
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
		os.environ['CUDA_VISIBLE_DEVICES'] = ''	#deactivate gpu

	# create directories if necessary
	if (os.path.isdir(model_dir)==0):
		os.mkdir(model_dir)
	if (os.path.isdir(result_dir)==0):
		os.mkdir(result_dir)

	is_training = FLAGS.is_training
	geometric_object = FLAGS.geometric_object
	model_id = FLAGS.model_id
	regularisation_type = FLAGS.regularisation_type
	lambda_reg = FLAGS.lambda_reg
	epochs = FLAGS.epochs

	dataset_dir_train = '/home/alasdair/ownCloud/Codes/Autoencoders/Geometric_autoencoder_keras/data/'+'train_'+geometric_object+'/'
	dataset_dir_test = '/home/alasdair/ownCloud/Codes/Autoencoders/Geometric_autoencoder_keras/data/'+'test_'+geometric_object+'/'
	if (is_training == 1):
		train_ae(dataset_dir_train,dataset_dir_test,model_id,geometric_object,regularisation_type,lambda_reg,epochs)
	else:
		test_ae(dataset_dir_test,model_id)