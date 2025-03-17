import torch
from train_validate import train_and_validate
from dataset import train_dataloader, val_dataloader
from model import initialize_model
from config import device, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY

# Initialize model and optimizer
model = initialize_model()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

# Start training and validation
train_and_validate(model, train_dataloader, val_dataloader, optimizer)