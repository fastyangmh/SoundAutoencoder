# import
import torch
import torch.nn as nn

# class


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm1d(num_features=8),
                                     nn.ReLU(),
                                     nn.Conv1d(
                                         in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm1d(num_features=16),
                                     nn.ReLU(),
                                     nn.Conv1d(
                                         in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm1d(num_features=32),
                                     nn.ReLU(),
                                     nn.Conv1d(
                                         in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm1d(num_features=64),
                                     nn.ReLU())
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm1d(num_features=32),
                                     nn.ReLU(),
                                     nn.ConvTranspose1d(
                                         in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm1d(num_features=16),
                                     nn.ReLU(),
                                     nn.ConvTranspose1d(
                                         in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm1d(num_features=8),
                                     nn.ReLU(),
                                     nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False))
        self.classifier = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
                                        nn.BatchNorm1d(num_features=32),
                                        nn.ReLU(),
                                        nn.Conv1d(
                                            in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False),
                                        nn.BatchNorm1d(num_features=1),
                                        nn.ReLU(),
                                        Flatten(),
                                        nn.Linear(in_features=625, out_features=6))

    def forward(self, x, y):
        code = self.encoder(x)
        y_hat = self.classifier(code)
        _, channels, _ = code.shape
        code = torch.cat(
            (code, torch.cat([y[:, None]]*channels, 1)[:, :, None]), -1)
        output = self.decoder(code)
        return output, y_hat


if __name__ == '__main__':
    # create model
    model = AutoEncoder()

    # create input data
    x = torch.rand(32, 1, 40000)
    y = torch.randint(0, 6, (32,))

    # get model output
    x_hat, y_hat = model(x, y)

    # display the dimension of input and output
    print(x.shape)
    print(x_hat[:, :, :-1].shape)
    print(y.shape)
    print(y_hat.shape)
