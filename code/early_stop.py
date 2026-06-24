import copy


class EarlyStop:
	def __init__(self, patience = 5, m_delta = 0.0):
		self.patience = patience
		self.m_delta = m_delta
		self.count = 0
		self.best_mse = float('inf')
		self.best_model = None

	def stop_early(self, validation_mse, model):
		if validation_mse < self.best_mse - self.m_delta:
			self.best_mse = validation_mse
			self.count = 0
			self.best_model = copy.deepcopy(model.state_dict())
		else:
			self.count += 1
			if self.count > self.patience:
				return True
		return False

	def restore_best(self, model):
		if self.best_model is not None:
			model.load_state_dict(self.best_model)