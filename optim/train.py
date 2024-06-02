import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import SpectrogramTripletDataset
from model import CNN
from optim.functional import cosine_schedule

def train_contrastive(
        model,
        optimiser,
        train_dataset,
        val_dataset,
        batch_size,
        num_epochs,
        config,
        log_dir=None
):
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler()

    # ============================== Data Handling ==============================
    # Initialise dataloaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #============================== Model Learning Parameters ==============================
    # LR schedule, warmup then cosine
    base_lr = optimiser.param_groups[0]['lr'] * batch_size / 256
    end_lr = 1e-6
    warm_up_lrs = torch.linspace(0, base_lr, 11)[1:]
    cosine_lrs = cosine_schedule(base_lr, end_lr, num_epochs-10)
    lrs = torch.cat([warm_up_lrs, cosine_lrs])
    assert len(lrs) == num_epochs

    # WD schedule, cosine 
    start_wd = 0.04
    end_wd = 0.4
    wds = cosine_schedule(start_wd, end_wd, num_epochs)

    if log_dir is not None:
        writer = SummaryWriter(log_dir=log_dir)
        # write config
        writer.add_text('config', str(config))
        writer.add_text('model', str(model))
    # Train the model
    for epoch in range(num_epochs):
        model.train()

        # Update lr
        for param_group in optimiser.param_groups:
            param_group['lr'] = lrs[epoch].item()
        # Update wd
        for param_group in optimiser.param_groups:
            if param_group['weight_decay'] != 0:
                param_group['weight_decay'] = wds[epoch].item()

        epoch_train_losses = []
        epoch_val_losses = []
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        for one, two, three in loop:

            with torch.cuda.amp.autocast():
                # Forward pass
                emb1, emb2, emb3 = model(one), model(two), model(three)
                sim_loss = F.mse_loss(emb1, emb2)
                diff_loss = 0.5 * F.mse_loss(emb1, emb3) + 0.5 * F.mse_loss(emb2, emb3)
                loss = max(sim_loss + config['margin'] - diff_loss, torch.tensor(0, device=device))

            # Backward pass and optimization
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            epoch_train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            for one, two, three in val_loader:
                with torch.cuda.amp.autocast():
                    emb1, emb2, emb3 = model(one), model(two), model(three)
                    sim_loss = F.mse_loss(emb1, emb2)
                    diff_loss = 0.5 * F.mse_loss(emb1, emb3) + 0.5 * F.mse_loss(emb2, emb3)
                    loss = max(sim_loss + config['margin'] - diff_loss, torch.tensor(0, device=device))

                epoch_val_losses.append(loss.item())
        
        if log_dir is not None:
            writer.add_scalar('Loss/train', sum(epoch_train_losses) / len(epoch_train_losses), epoch)
            writer.add_scalar('Loss/val', sum(epoch_val_losses) / len(epoch_val_losses), epoch)