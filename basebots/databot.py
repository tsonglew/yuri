class DataBot:
    def __init__(self):
        self.train_data = list()

    def append_data(self, data):
        if data is not None:
            self.train_data.append(data)

    def get_data(self):
        return self.train_data
