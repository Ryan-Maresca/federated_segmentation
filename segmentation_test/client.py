import flwr as fl
from seg_models import build_unet
from data_loader import demo_train_val_test, show_predictions


class FlwrClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        hist = model.fit(train, epochs=2, verbose=2)
        print(f'hist loss: {hist.history["loss"]}')
        return model.get_weights(), len(train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc = model.evaluate(test, verbose=2)
        return loss, len(test), {"accuracy": acc}

