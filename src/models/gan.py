import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, feature_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(generator, discriminator, real_data_loader, noise_dim, epochs=50, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, discriminator = generator.to(device), discriminator.to(device)
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for real_data in real_data_loader:
            real_data = real_data[0].to(device)
            batch_size = real_data.size(0)
            # Train Discriminator
            z = torch.randn(batch_size, noise_dim).to(device)
            fake_data = generator(z)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            d_loss_real = criterion(discriminator(real_data), real_labels)
            d_loss_fake = criterion(discriminator(fake_data.detach()), fake_labels)
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
            # Train Generator
            g_loss = criterion(discriminator(fake_data), real_labels)
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}") 