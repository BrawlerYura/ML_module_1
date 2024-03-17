import fire
from BrawlerClassifierModel import BrawlerClassifierModel


class CLI(object):

    def __init__(self):
        self.model = BrawlerClassifierModel()

    def train(self, dataset):
        return self.model.train(dataset)

    def predict(self, dataset):
        return self.model.predict(dataset)


if __name__ == "__main__":
    fire.Fire(CLI)
