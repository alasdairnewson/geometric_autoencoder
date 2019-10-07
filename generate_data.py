

from image_utils import *


def monte_carlo_shape(params,img_size):

	h_size = ( int(np.floor(img_size[0]/2.0)) , int(np.floor(img_size[1]/2.0)))
	X,Y = np.meshgrid( range( -h_size[1],(h_size[1]+1) ) , range( -h_size[0], (h_size[0]+1) ) )

	img_out = np.zeros(img_size)
	for i in range(0,params['N_monte_carlo']):
		q = params['sigma'] * np.random.randn(2,1)
		X_temp = X+q[0]
		Y_temp = Y+q[1]
	
		# indicator functions for different shapes
		if (params['geometric_object'] == 'disk'):
			img_temp = (X_temp**2+Y_temp**2) < params['radius']**2
		elif(params['geometric_object'] == 'disk_shifted'):
			img_temp = (X_temp-params['x_center'])**2+(Y_temp-params['y_center'])**2 < params['radius']**2
		elif(params['geometric_object'] == 'circle'):
			img_temp_1 = (X_temp**2 + Y_temp**2) < params['radius']**2
			img_temp_2 = (X_temp**2 + Y_temp**2) < (params['radius']+1)**2
			img_temp = np.abs(img_temp_1.astype(float)-img_temp_2.astype(float))
		elif(params['geometric_object'] == 'square'):
			img_temp = np.maximum(np.abs(X_temp),np.abs(Y_temp)) < params['radius']
		elif (params['geometric_object'] == 'disk_shifted'):
			img_temp = ((X_temp-params['x_shift'])**2+(Ytemp-params['y_shift'])**2) < params['radius']
		elif (params['geometric_object'] == 'ellipse'):
			a = params['x_radius']
			b = params['y_radius']
			theta = params['theta']
			A = np.asarray([ [np.cos(-theta) , -np.sin(-theta)] , [np.sin(-theta) , np.cos(-theta) ]])  #carry out the inverse transformation to align the axes
			
			coords_transformed = A @ np.vstack((X_temp.flatten() , Y_temp.flatten()))
			X_temp = np.reshape(coords_transformed[0,:],img_size)
			Y_temp = np.reshape(coords_transformed[1,:],img_size)
			
			img_temp = ( ((X_temp/a)**2 + (Y_temp/b)**2 ) <= 1.0).astype(float)
		else:
			print('Error, unknown shape')
		
		img_out = img_out + img_temp

	img_out = img_out/(float(params['N_monte_carlo']))

	return img_out

def generate_data(geometric_object,img_size=(63,63), n_images_train = 1000,n_images_test=300):

	# blurring parameter
	sigma = 0.5
	N_monte_carlo = 800
	data_dir = 'data/'
	if (os.path.isdir(data_dir)==0):
		os.mkdir(data_dir)

	params = {'geometric_object':geometric_object,'sigma':sigma,'N_monte_carlo':N_monte_carlo}

	# create train and test data folders if necessary
	train_data_dir = data_dir+'train_'+geometric_object+"/"
	if (os.path.isdir(train_data_dir)==0):
		os.mkdir(train_data_dir)
	test_data_dir = data_dir+'test_'+geometric_object+"/"
	if (os.path.isdir(test_data_dir)==0):
		os.mkdir(test_data_dir)

	# generate training data
	max_radius = np.min(np.asarray(img_size))/(2.0)
	for i in range(0,n_images_train):
		
		if (params['geometric_object'] == 'disk' or params['geometric_object'] == 'square' or params['geometric_object'] == 'circle'):
			rand_radius = max_radius*np.random.rand()
			params['radius'] = rand_radius
		elif  (params['geometric_object'] == 'ellipse'):
			a = max_radius*np.random.rand()
			b = max_radius*np.random.rand()
			
			params['x_radius'] = a
			params['y_radius'] = b
			params['theta'] = (np.pi/2.0)*(np.random.rand()-0.5)
		
		img_out = monte_carlo_shape(params, img_size)
		img_out = np.tile( np.expand_dims(img_out,axis=2) , (1,1,3) )
		write_image(img_out,train_data_dir+geometric_object+'_number_'+str(i).zfill(6)+'.png')

	# generate test data
	if (params['geometric_object'] == 'disk' or params['geometric_object'] == 'square' or params['geometric_object'] == 'circle'):
		radius_list = np.linspace(0,max_radius,n_images_test)
		for i in range(0,len(radius_list)):
			radius = radius_list[i]
			params['radius'] = radius
			img_out = monte_carlo_shape(params, img_size)
			write_image(img_out,test_data_dir+params['geometric_object'] + '_number_' + str(i).zfill(6) + '_radius_' + str(int(np.round(radius))).zfill(6) + '.png')
	elif (params['geometric_object'] == 'ellipse'):
	
		# create list of parameters
		min_radius = 1.0
		x_radius_list = np.linspace(min_radius,max_radius,n_images_test**(1.0/3.0))
		y_radius_list = np.linspace(min_radius,max_radius,n_images_test**(1.0/3.0))
		theta_list = np.linspace(-np.pi/4.0,np.pi/4.0,n_images_test**(1.0/3.0))
		for i in range(0,len(x_radius_list)):
			for j in range(0,len(y_radius_list)):
				for k in range(0,len(y_radius_list)):
					a = x_radius_list[i]
					b = y_radius_list[j]
					
					params['x_radius'] = a
					params['y_radius'] = b
					params['theta'] = theta_list[k]
					theta_degrees = params['theta']*180.0/(np.pi)
					img_out = monte_carlo_shape(params, img_size)
					write_image(img_out,test_data_dir+params['geometric_object'] +\
						'_number_' + str(i).zfill(6) + '_x_radius_' + str(int(np.round(a))).zfill(6) +\
						'_y_radius_' + str(int(np.round(b))).zfill(6) + '_theta_' + str(int(np.round(theta_degrees))).zfill(6) + '.png')


if __name__ == '__main__':

	flags = tf.app.flags
	flags.DEFINE_string("geometric_object", "disk", "Geometric object for data [Disk]")
	FLAGS = flags.FLAGS

	img_size = (63,63)
	geometric_object = FLAGS.geometric_object

	generate_data(geometric_object,img_size)