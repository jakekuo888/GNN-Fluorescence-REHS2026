import torch
import torch.nn as nn
import copy


class EarlyStop:
	def __init__(self, patience = 5, m_delta = 0.0):
		self.patience = patience
		self.m_delta = m_delta
		self.count = 0
		self.best_mse = float('inf')
		#NOTE: have it copy the weights when early stop.
		#This is incomplete.

	def stop_early(self, validation_mse):
		if validation_mse < self.best_mse - self.m_delta:
			self.best_mse = validation_mse
			self.count = 0
		else:
			self.count += 1
			if self.count > self.patience:
				return True
		return False