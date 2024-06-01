import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def train(
        model,
        optimizer,
        train_loader,
        val_loader,
        num_epochs,
        log_dir=None
):
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
    # Train the model
    for epoch in range(num_epochs):
        model.train()

        epoch_train_losses = []
        epoch_val_losses = []
        for one, two, three in train_loader:
            one, two, three = one.to(device), two.to(device), three.to(device)

            with torch.cuda.amp.autocast():
                # Forward pass
                emb1, emb2, emb3 = model(one), model(two), model(three)
                loss = loss_fn(emb1, emb2, emb3)

            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            for one, two, three in val_loader:
                one, two, three = one.to(device), two.to(device), three.to(device)

                with torch.cuda.amp.autocast():
                    emb1, emb2, emb3 = model(one), model(two), model(three)
                    loss = loss_fn(emb1, emb2, emb3)

                epoch_val_losses.append(loss.item())
        
        if log_dir is not None:
            writer.add_scalar('Loss/train', sum(epoch_train_losses) / len(epoch_train_losses), epoch)
            writer.add_scalar('Loss/val', sum(epoch_val_losses) / len(epoch_val_losses), epoch)