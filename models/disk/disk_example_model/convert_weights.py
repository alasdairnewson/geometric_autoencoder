
import pdb
import pickle

with open('disk_example_model.pickle', 'rb') as handle:
	parameter_dict = pickle.load(handle)

#training_object_type = parameter_dict.pop('training_object')
parameter_dict['model_id'] = 'disk_example_model'
with open('disk_example_model.pickle', 'wb') as handle:
		pickle.dump(parameter_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# dictionary[new_key] = dictionary[old_key]
# del dictionary[old_key]