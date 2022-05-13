import os
import time
from statistics import mean

import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from model import StackedAutoencoder

NUM_EPOCHS = 100
verbose_printing = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def main():
	trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
	testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

	train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
	test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

	model = StackedAutoencoder(input_size=3, output_size=64, verbose_printing=verbose_printing)
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0001)
	criterion = torch.nn.CrossEntropyLoss()

	model = model.to(device)
	print("-"*156 if verbose_printing else "-"*88)

	log_string = "[{:3}] Time: {:.2f}\tTrain Loss: {:.4f}\tVal Loss: {:.4f}\tAccuracy: {:.2f}%"
	if verbose_printing:
		log_string = "[{:3}] Time: {:.2f}\tTrain Loss: {:.4f}({:.4f}, {:.4f}, {:.4f})\tVal Loss: {:.4f}({:.4f}, {:.4f}, {:.4f})\tAccuracy: {:.2f}%({:.2f}, {:.2f}, {:.2f})"
	
	for epoch in range(1, NUM_EPOCHS + 1):
		start_time = time.time()
		train_loss = train(train_loader, model, criterion, optimizer, verbose_printing)
		val_loss, val_accuracy = validate(test_loader, model, criterion, verbose_printing)

		if verbose_printing:
			train_loss, tloss1, tloss2, tloss3 = train_loss
			val_loss, vloss1, vloss2, vloss3 = val_loss
			val_accuracy, vacc1, vacc2, vacc3 = val_accuracy

			print(log_string.format(epoch, time.time() - start_time, train_loss, tloss1, tloss2, tloss3, val_loss, vloss1, vloss2, vloss3, val_accuracy, vacc1, vacc2, vacc3))
		else:
			print(log_string.format(epoch, time.time() - start_time, train_loss, val_loss, val_accuracy))

def train(train_loader, model, criterion, optimizer, verbose_printing=False):
	train_losses, loss1, loss2, loss3 = [], [], [], []

	model.train()
	for batch_idx, (images, labels) in enumerate(train_loader):
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		images = Variable(images)
		labels = Variable(labels)

		outputs = model(images, labels)

		if verbose_printing:
			outputs, y1, y2, y3 = outputs
			loss1.append(criterion(y1, labels).item())
			loss2.append(criterion(y2, labels).item())
			loss3.append(criterion(y3, labels).item())

		loss = criterion(outputs, labels)

		loss.backward()
		optimizer.step()

		train_losses.append(loss.item())

	return_loss = mean(train_losses)

	if verbose_printing:
		return_loss = (return_loss, mean(loss1), mean(loss2), mean(loss3))

	return return_loss

def accuracy(outputs, labels):
	_, predicted = torch.max(outputs.data, 1)
	total = labels.size(0)
	correct = (predicted == labels).sum().item()
	return total, correct

def validate(val_loader, model, criterion, verbose_printing=False):
	val_losses, loss1, loss2, loss3 = [], [], [], []
	total = 0
	correct, correct1, correct2, correct3 = 0, 0, 0, 0

	model.eval()
	for batch_idx, (images, labels) in enumerate(val_loader):
		images, labels = images.to(device), labels.to(device)
		with torch.no_grad():
			images = Variable(images)
			labels = Variable(labels)

		outputs = model(images, labels)

		if verbose_printing:
			outputs, y1, y2, y3 = outputs
			loss1.append(criterion(y1, labels).item())
			loss2.append(criterion(y2, labels).item())
			loss3.append(criterion(y3, labels).item())

		loss = criterion(outputs, labels)

		total += labels.size(0)

		_, predicted = torch.max(outputs.data, 1)
		correct += (predicted == labels).sum().item()

		if verbose_printing:
			_, predicted = torch.max(y1.data, 1)
			correct1 += (predicted == labels).sum().item()

			_, predicted = torch.max(y2.data, 1)
			correct2 += (predicted == labels).sum().item()

			_, predicted = torch.max(y3.data, 1)
			correct3 += (predicted == labels).sum().item()

		val_losses.append(loss.item())
	
	return_loss = mean(val_losses)
	return_acc = 100 * correct / total
	if verbose_printing:
		return_loss = (return_loss, mean(loss1), mean(loss2), mean(loss3))
		return_acc = (return_acc, (100 * correct1 / total), (100 * correct2 / total), (100 * correct3 / total))

	return return_loss, return_acc

if __name__ == "__main__":
	main()