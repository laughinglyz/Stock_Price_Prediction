import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error,accuracy_score

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def MAPE(prediction, true):
	return np.mean(np.abs((true-prediction)/true)) * 100

def run_model(model,scaler, running_mode='train', train_set=None, valid_X=None, valid_Y=None, test_X=None, test_Y=None,
	batch_size=8, learning_rate=1e-4, n_epochs=40, stop_thr=1e-6, shuffle=True):
	
	if running_mode == 'train':
		train_loss, valid_loss, valid_RMSE, valid_MAPE, valid_accuracy = [], [], [], [], []
		trainloader = DataLoader(train_set,batch_size=batch_size,shuffle=shuffle)
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		last_loss, loss_change, e_count = float('inf'),float('inf'), 0
		while (n_epochs-e_count):
			model, tl = _train(model,trainloader,optimizer)
			train_loss.append(tl)
			cur_loss,RMSE,MAPE,accuracy = _test(model, valid_X, valid_Y, scaler)
			valid_loss.append(cur_loss)
			valid_RMSE.append(RMSE)
			valid_MAPE.append(MAPE)
			valid_accuracy.append(accuracy)
			loss_change = last_loss- cur_loss
			last_loss = cur_loss
			e_count+=1
		return model, train_loss, valid_loss, valid_RMSE, valid_MAPE, valid_accuracy, e_count
	else:
		return _test(model, test_X, test_Y, scaler)

def _train(model,data_loader,optimizer, device=torch.device('cpu')):
	model.to(device)
	model.train()
	criterion = nn.MSELoss()
	train_loss = []
	for data in data_loader:
		inputs, labels = data
		inputs.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs,labels.float())
		loss.backward()
		optimizer.step()
		train_loss.append(loss.item())

	return model, np.mean(train_loss)


def _test(model, test_X, test_Y, scaler, device=torch.device('cpu')):
	model.to(device)
	model.eval()
	inputs = torch.from_numpy(test_X)
	inputs.to(device)
	with torch.no_grad():
		outputs = model(inputs).numpy()
		loss = mean_squared_error(outputs,scaler.transform(test_Y.reshape(-1,1)).reshape(-1,))
		outputs = scaler.inverse_transform(outputs.reshape(-1,1)).reshape(-1,)
		test_RMSE = np.sqrt(mean_squared_error(outputs,test_Y))
		test_MAPE = MAPE(outputs,test_Y)
		St = scaler.inverse_transform(test_X[:,-1,0].reshape(-1,1)).reshape(-1,)
		directional_true,directional_prediction = np.ones(test_X.shape[0]),np.ones(test_X.shape[0])
		directional_true[St>=test_Y] = -1
		directional_prediction[St>=outputs] = -1
		test_accuracy = accuracy_score(directional_true, directional_prediction)

		plt.xlabel("time")
		plt.ylabel("HSI value")
		plt.plot(range(len(outputs)),outputs,label="prediction")
		plt.plot(range(len(outputs)),test_Y,label="actual")
		plt.legend()
		plt.show()
	return loss, test_RMSE, test_MAPE, test_accuracy
	




