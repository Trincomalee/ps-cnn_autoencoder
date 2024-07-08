from torch import nn


# Die Anzahl der Neuronen in der letzten Schicht vom Encoder bezeichnen wir mit "Engpass",
# da diese Schicht besitzt normalerweise die wenigsten Neuronen.
########################################################################################################################
##################### ursprüngliche Architektur #####################
########################################################################################################################
class Autoencoder_original(nn.Module):
    def __init__(self, engpass=32):
        super(Autoencoder_original, self).__init__()

        self.encoder = nn.Sequential(
            # encode1
            nn.Linear(784, 128, bias=True),
            nn.ReLU(),
            # encode2
            nn.Linear(128, 64, bias=True),
            nn.Tanh(),
            # encode3
            nn.Linear(64, engpass, bias=True),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # decode3
            nn.Linear(engpass, 64, bias=True),
            nn.ReLU(),
            # decode2
            nn.Linear(64, 128, bias=True),
            nn.Sigmoid(),
            # decode1
            nn.Linear(128, 784, bias=True),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



########################################################################################################################
##################### modifizierte Architektur #####################
########################################################################################################################
# Verändert ist nur die Wahl von Aktivierungsfunktion.
class Autoencoder_modified(nn.Module):
    def __init__(self, engpass=32):
        super(Autoencoder_modified, self).__init__()


        self.encoder = nn.Sequential(
            # encode1:
            nn.Linear(784, 128, bias=True),
            nn.Sigmoid(),
            # encode2
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            # encode3
            nn.Linear(64, engpass, bias=True),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            # decode3
            nn.Linear(engpass, 64, bias=True),
            nn.ReLU(),
            # decode2
            nn.Linear(64, 128, bias=True),
            nn.ReLU(),
            # decode1
            nn.Linear(128, 784, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


