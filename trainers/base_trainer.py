class BaseTrainer:

    def __init__(self, config_json):
        self.name = 'BaseTrainer'

        self.test_size = config_json.get('test_size')
        self.batch_size = config_json.get('batch_size')
        self.learning_rate = config_json.get('learning_rate')
        self.hm_epochs = config_json.get('hm_epochs')
        self.epochs = config_json.get('epochs')
        self.increment = config_json.get('increment')

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
        self.model.fit(x_train, y_train, x_test, y_test, self.epochs, self.batch_size)
        return self

    def save(self, save2path):
        self.model.save(save2path)
        return self

    def load(self, model):
        self.model.load(model)
        return self
