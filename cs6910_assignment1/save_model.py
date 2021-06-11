from main import *
import pickle


network=master(batch=495, epochs=7, output_dim=10, activation='tanh', opt=ADAM(eta=0.003576466933615937,layers=4,weight_decay=0.31834976996809683), layer_1 = 32,layer_2 =64 ,layer_3=16 ,weight_init='xavier',loss_type='cross_entropy',augment=100)
print(len(network))
print(network)
filename_model = 'neural_network.object'
pickle.dump(network, open(filename_model, 'wb'))  # store best model's  object to disk