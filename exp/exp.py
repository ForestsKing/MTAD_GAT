import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MyDataset
from model.loss import JointLoss
from model.mtad_gat import MTAD_GAT
from utils.adjustpred import adjust_predicts
from utils.earlystop import EarlyStop
from utils.evalmethods import pot_threshold, epsilon_threshold, bestf1_threshold
from utils.plot import plot_loss
from utils.preprocess import preprocess


class Exp:
    def __init__(self, group, iter, epochs, batch_size, patience, lr, generate, w=64, gamma=1):
        self.group = group

        self.iter = iter
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.w = w
        self.gamma = gamma
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = './checkpoint/' + self.group + '_checkpoint_iter' + str(self.iter) + '.pkl'
        self._get_data(generate=generate)
        self._get_model()

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')
        if not os.path.exists('./img/'):
            os.makedirs('./img/')
        if not os.path.exists('./result/'):
            os.makedirs('./result/')

    def _get_data(self, generate, train=True):
        if train:
            self.train_x, self.valid_x, self.test_x, self.test_y = preprocess(generate=generate, group=self.group)
            trainset = MyDataset(self.train_x, w=self.w)
            validset = MyDataset(self.valid_x, w=self.w)
            testset = MyDataset(self.test_x, w=self.w)

            self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
            self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True)
            self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=True)

            self.loss = {'train': {'forecast': [], 'reconstruct': [], 'total': []},
                         'valid': {'forecast': [], 'reconstruct': [], 'total': []}}

            print('train: {0}, valid: {1}, test: {2}'.format(len(trainset), len(validset), len(testset)))
        else:
            self.train_x, self.valid_x, self.test_x, self.test_y = preprocess(generate=generate, group=self.group)
            self.train_x = np.vstack((self.train_x, self.valid_x))

            trainset = MyDataset(self.train_x, w=self.w)
            testset = MyDataset(self.test_x, w=self.w)

            self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
            self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=True)

            print('train: {0}, test: {1}'.format(len(trainset), len(testset)))

    def _get_model(self):
        self.model = MTAD_GAT().to(self.device)
        self.criterion = JointLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.earlystopping = EarlyStop(patience=self.patience)

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        reconstruct, forecast = self.model(batch_x)
        forecast_loss, reconstruct_loss, loss = self.criterion(batch_x, batch_y, reconstruct, forecast)

        return forecast_loss, reconstruct_loss, loss

    def _get_score(self, data, dataloader):
        self.model.eval()
        forecasts, reconstructs = [], []
        for (batch_x, batch_y) in tqdm(dataloader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            _, forecast = self.model(batch_x)

            recon_x = torch.cat((batch_x[:, 1:, :], batch_y), dim=1)
            reconstruct, _ = self.model(recon_x)

            forecasts.append(forecast.detach().cpu().numpy())
            reconstructs.append(reconstruct.detach().cpu().numpy()[:, -1, :])

        forecasts = np.concatenate(forecasts, axis=0).squeeze()
        reconstructs = np.concatenate(reconstructs, axis=0)
        actuals = data[self.w:]

        df = pd.DataFrame()
        scores = np.zeros_like(actuals)
        for i in range(actuals.shape[1]):
            df["For_" + str(i)] = forecasts[:, i]
            df["Rec_" + str(i)] = reconstructs[:, i]
            df["Act_" + str(i)] = actuals[:, i]

            score = np.sqrt((forecasts[:, i] - actuals[:, i]) ** 2) + self.gamma * np.sqrt(
                (reconstructs[:, i] - actuals[:, i]) ** 2)
            scores[:, i] = score
            df["Score_" + str(i)] = score

        scores = np.mean(scores, axis=1)
        df['Score_Global'] = scores

        return df

    def fit(self):
        # init loss
        self.model.eval()
        train_forecast_loss, train_reconstruct_loss, train_loss = [], [], []
        for (batch_x, batch_y) in tqdm(self.trainloader):
            forecast_loss, reconstruct_loss, loss = self._process_one_batch(batch_x, batch_y)
            train_forecast_loss.append(forecast_loss.item())
            train_reconstruct_loss.append(reconstruct_loss.item())
            train_loss.append(loss.item())

        self.model.eval()
        valid_forecast_loss, valid_reconstruct_loss, valid_loss = [], [], []
        for (batch_x, batch_y) in self.validloader:
            forecast_loss, reconstruct_loss, loss = self._process_one_batch(batch_x, batch_y)
            valid_forecast_loss.append(forecast_loss.item())
            valid_reconstruct_loss.append(reconstruct_loss.item())
            valid_loss.append(loss.item())

        train_forecast_loss = np.sqrt(np.average(np.array(train_forecast_loss) ** 2))
        valid_forecast_loss = np.sqrt(np.average(np.array(valid_forecast_loss) ** 2))
        train_reconstruct_loss = np.sqrt(np.average(np.array(train_reconstruct_loss) ** 2))
        valid_reconstruct_loss = np.sqrt(np.average(np.array(valid_reconstruct_loss) ** 2))
        train_loss = np.sqrt(np.average(np.array(train_loss) ** 2))
        valid_loss = np.sqrt(np.average(np.array(valid_loss) ** 2))

        print(
            "Iter: {0} Init || Total Loss| Train: {1:.6f} Vali: {2:.6f} || Forecast Loss| Train:{3:.6f} Valid"
            ": {4:.6f} || Reconstruct Loss| Train: {5:.6f} Valid: {6:.6f}".format(
                self.iter, train_loss, valid_loss, train_forecast_loss, valid_forecast_loss,
                train_reconstruct_loss, valid_reconstruct_loss))

        for e in range(self.epochs):
            self.model.train()
            train_forecast_loss, train_reconstruct_loss, train_loss = [], [], []
            for (batch_x, batch_y) in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                forecast_loss, reconstruct_loss, loss = self._process_one_batch(batch_x, batch_y)
                train_forecast_loss.append(forecast_loss.item())
                train_reconstruct_loss.append(reconstruct_loss.item())
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            valid_forecast_loss, valid_reconstruct_loss, valid_loss = [], [], []
            for (batch_x, batch_y) in self.validloader:
                forecast_loss, reconstruct_loss, loss = self._process_one_batch(batch_x, batch_y)
                valid_forecast_loss.append(forecast_loss.item())
                valid_reconstruct_loss.append(reconstruct_loss.item())
                valid_loss.append(loss.item())

            train_forecast_loss = np.sqrt(np.average(np.array(train_forecast_loss) ** 2))
            valid_forecast_loss = np.sqrt(np.average(np.array(valid_forecast_loss) ** 2))
            train_reconstruct_loss = np.sqrt(np.average(np.array(train_reconstruct_loss) ** 2))
            valid_reconstruct_loss = np.sqrt(np.average(np.array(valid_reconstruct_loss) ** 2))
            train_loss = np.sqrt(np.average(np.array(train_loss) ** 2))
            valid_loss = np.sqrt(np.average(np.array(valid_loss) ** 2))

            self.loss['train']['forecast'].append(train_forecast_loss)
            self.loss['train']['reconstruct'].append(train_reconstruct_loss)
            self.loss['train']['total'].append(train_loss)
            self.loss['valid']['forecast'].append(valid_forecast_loss)
            self.loss['valid']['reconstruct'].append(valid_reconstruct_loss)
            self.loss['valid']['total'].append(valid_loss)

            print(
                "Iter: {0} Epoch: {1} || Total Loss| Train: {2:.6f} Vali: {3:.6f} || Forecast Loss| Train:{4:.6f} Valid"
                ": {5:.6f} || Reconstruct Loss| Train: {6:.6f} Valid: {7:.6f}".format(
                    self.iter, e+1, train_loss, valid_loss, train_forecast_loss, valid_forecast_loss,
                    train_reconstruct_loss, valid_reconstruct_loss))

            self.earlystopping(valid_loss, self.model, self.path)
            if self.earlystopping.early_stop:
                print("Iter {0} is Early stopping!".format(self.iter))
                break
        self.model.load_state_dict(torch.load(self.path))

        plot_loss(self.loss["train"]["forecast"], self.loss["train"]["reconstruct"], self.loss["train"]["total"],
                  './img/' + str(self.group) + '_iter' + str(self.iter) + '_trainloss.png')
        plot_loss(self.loss["valid"]["forecast"], self.loss["valid"]["reconstruct"], self.loss["valid"]["total"],
                  './img/' + str(self.group) + '_iter' + str(self.iter) + '_validloss.png')

    def predict(self, model_load=False, data_load=False):
        if model_load:
            self.model.load_state_dict(torch.load(self.path))
        self._get_data(generate=False, train=False)

        actual_label = self.test_y[self.w:]

        if data_load:
            trainresult = pd.read_csv('./result/' + str(self.group) + '_iter' + str(self.iter) + '_trainresult.csv')
            testresult = pd.read_csv('./result/' + str(self.group) + '_iter' + str(self.iter) + '_testresult.csv')
        else:
            trainresult = self._get_score(self.train_x, self.trainloader)
            testresult = self._get_score(self.test_x, self.testloader)

        for i in range(self.test_x.shape[1]):
            train_score = trainresult["Score_" + str(i)].values
            test_score = testresult["Score_" + str(i)].values

            threshold = pot_threshold(train_score, test_score)

            train_pred = (train_score > threshold).astype(np.int)
            test_pred = (test_score > threshold).astype(np.int)

            trainresult["Pred_" + str(i)] = train_pred
            trainresult["Threshold_" + str(i)] = threshold
            testresult["Pred_" + str(i)] = test_pred
            testresult["Threshold_" + str(i)] = threshold

        train_score = trainresult["Score_Global"].values
        test_score = testresult["Score_Global"].values

        # threshold = pot_threshold(train_score, test_score)
        threshold = bestf1_threshold(test_score, actual_label)
        # threshold = epsilon_threshold(train_score)

        train_pred = (train_score > threshold).astype(np.int)
        test_pred = (test_score > threshold).astype(np.int)

        train_pred = adjust_predicts(np.zeros_like(train_pred), train_pred)
        test_pred = adjust_predicts(actual_label, test_pred)

        trainresult["Pred_Global"] = train_pred
        trainresult["Label_Global"] = 0
        trainresult["Threshold_Global"] = threshold

        testresult["Pred_Global"] = test_pred
        testresult["Label_Global"] = actual_label
        testresult["Threshold_Global"] = threshold

        trainresult.to_csv('./result/' + str(self.group) + '_iter' + str(self.iter) + '_trainresult.csv', index=False)
        testresult.to_csv('./result/' + str(self.group) + '_iter' + str(self.iter) + '_testresult.csv', index=False)

        print("Iter {0} Group {1} || precision: {2:.6f} recall: {3:.6f} f1: {4:.6f}".format(
            self.iter, self.group, precision_score(actual_label, test_pred),
            recall_score(actual_label, test_pred), f1_score(actual_label, test_pred)))
