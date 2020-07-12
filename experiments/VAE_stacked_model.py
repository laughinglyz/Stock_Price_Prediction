from code import load_daily_stock_price
from code import preprocess_VAE, HSI_Dataset
from code import run_model, VAE
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.ensemble import *
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.preprocessing import MinMaxScaler

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


def loss_function(recon_x, x, mu, logvar):
    mse = nn.MSELoss(reduction='sum')
    loss = mse(recon_x,x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)/128
    return loss + KLD

df = load_daily_stock_price()
df = df.dropna()
train_X, train_Y, test_X, test_Y,scaler = preprocess_VAE(df)

train_set = HSI_Dataset(train_X,train_Y)
dataloader = DataLoader(train_set,batch_size=128,shuffle=True)
num_epochs = 100

model = VAE(input_dim=1, hidden_size=128, z_size=20)
model = model.float()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.train()
L =[]
for epoch in range(num_epochs):
    train_loss = 0
    for X, Y in dataloader:
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(X)
        loss = loss_function(recon_batch, X.float(), mu, logvar)
        loss.backward()
        optimizer.step()
        L.append(loss.item())

model.eval()
with torch.no_grad():
    train_mu, train_logvar = model.encode(torch.from_numpy(train_X))
    train_features = model.reparametrize(train_mu,train_logvar).numpy()
    test_mu, test_logvar = model.encode(torch.from_numpy(test_X))
    test_features = model.reparametrize(test_mu,test_logvar).numpy()
    train_pred, _ , _ = model(torch.from_numpy(train_X))
    train_pred = train_pred.numpy()
    plt.plot(range(train_X.shape[0]),train_X,label="train_X")
    plt.plot(range(train_X.shape[0]),train_pred,label="fit train_X")
    plt.legend()
    plt.show()
    print(train_features.shape)
    print(test_features.shape)


ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
bagging = BaggingRegressor(n_estimators=500)
et = ExtraTreesRegressor(n_estimators=500)
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=500)

losses,RMSEs,MAPEs,accuracies = [],[],[],[]

train_Y, test_Y = train_Y.reshape(-1,),test_Y.reshape(-1,)
inv_test_Y,inv_test_X = scaler.inverse_transform(test_Y.reshape(-1,1)).reshape(-1,),scaler.inverse_transform(test_X.reshape(-1,1)).reshape(-1,)
plt.plot(range(test_Y.shape[0]),inv_test_Y,label="real price")
model_names = ["AdaBoost","Bagging","ExtraTree","GradientBoosting","RandomForest"]
for i,model in enumerate([ada,bagging,et,gb,rf]):

    model.fit(train_features,train_Y)
    prediciton = model.predict(test_features)
    loss = mean_squared_error(test_Y,prediciton)
    inv_pred = scaler.inverse_transform(prediciton.reshape(-1,1)).reshape(-1,)

    test_RMSE = np.sqrt(mean_squared_error(inv_pred,inv_test_Y))
    test_MAPE = np.mean(np.abs((inv_test_Y-inv_pred)/inv_test_Y)) * 100
    directional_true,directional_prediction = np.ones(test_X.shape[0]),np.ones(test_X.shape[0])
    directional_true[inv_test_X>=inv_test_Y] = -1
    directional_prediction[inv_test_X>=inv_pred] = -1
    test_accuracy = accuracy_score(directional_true, directional_prediction)
    
    losses.append(loss)
    RMSEs.append(test_RMSE)
    MAPEs.append(test_MAPE)
    accuracies.append(test_accuracy)

    plt.plot(range(test_Y.shape[0]),inv_pred,label=str(model_names[i])+" prediction")


plt.legend()
plt.show()

plt.bar(model_names,losses)
plt.title("loss")
plt.show()

plt.bar(model_names,RMSEs)
plt.title("RMSE")
plt.show()

plt.bar(model_names,MAPEs)
plt.title("MAPE")
plt.show()

plt.bar(model_names,accuracies)
plt.title("Accuracy")
plt.show()