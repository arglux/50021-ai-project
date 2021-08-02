from model import Net, LinReg2

import torch
import pandas as pd


def train(train_data, model):
	model.eval()
	for _X, _y in test_dl:
		test_batch_losses = []
		_X = Variable(_X).cuda().float()
		_y = Variable(_y).cuda().float()

		#apply model
		test_preds = model(_X)
		test_loss = criterion(test_preds, _y)

		for i in range(100): print(math.floor(10**_y[i]-1),math.floor(10**test_preds[i].item()-1))
		break

def sensitivity_evaluation(model, test_data, test_label, device='cpu'):
	TP = 0
	TN = 0
	FN = 0
	FP = 0

	for i in range(0, test_data.size()[0]):
		# print(test_data[i].size())
		Xtest = torch.Tensor(test_data[i]).to(device)
		y_hat = model(Xtest)

		if y_hat > 0.5: prediction = 1
		else: prediction = 0

		if (prediction == test_label[i]):
			if (prediction == 1): TP += 1
			else: TN += 1

		else:
			if (prediction == 1): FP += 1
			else: FN += 1

	print("True Positives: {0}, True Negatives: {1}".format(TP, TN))
	print("False Positives: {0}, False Negatives: {1}".format(FP, FN))
	rate = TP/(FN+TP)
	print("Class specific accuracy of correctly predicting a hit song is {0}".format(rate))

if __name__ == '__main__':
	print("OK")
	# inp_size = 49
	# out_size = 1
	# device = 'cpu'

	# trained_model = Net(inp_size, out_size).to(device) # reinitialize model
	# trained_model.load_state_dict(torch.load('./models/dummy-model-2706-2356'))
	# trained_model.eval()

	# test_data = pd.read_csv('./data/herremans_hit_1030test.csv')
	# print(f"Data loaded. Train data shape: {test_data.shape}")

	# x = torch.FloatTensor(test_data.loc[:, test_data.columns != 'Topclass1030'].values).to(device)
	# y = torch.FloatTensor(test_data['Topclass1030']).to(device)

	# sensitivity_evaluation(trained_model, x, y)
