class BaseTrainer:

    def __init__(self):
        self.test_size = 100
        self.batch_size = 128
        self.learning_rate = 0.0001
        self.hm_epochs = 100
        self.increment = 200
        self.name = 'BaseTrainer'

    def prepare_model(self, model):
        """
        create a basic cnn model or reload a model from path if reuse
        """
        if model is not None:
            self.load(model)
        else:
            self.model.init()
        self.model.compile(lr=self.learning_rate)
        return self

    def fit(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train, x_test, y_test, self.batch_size)
        return self

    def save(self, save2path):
        self.model.save(save2path)
        return self

    def load(self, model):
        self.model.load(model)
        return self
