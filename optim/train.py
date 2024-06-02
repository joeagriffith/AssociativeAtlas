import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.dataset import SpectrogramTripletDataset
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
                loss = sim_loss - diff_loss

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
                    loss = sim_loss - diff_loss

                epoch_val_losses.append(loss.item())
        
        if log_dir is not None:
            writer.add_scalar('Loss/train', sum(epoch_train_losses) / len(epoch_train_losses), epoch)
            writer.add_scalar('Loss/val', sum(epoch_val_losses) / len(epoch_val_losses), epoch)

def train_classify(
        model,
        optimiser,
        train_dataset,
        val_dataset,
        batch_size,
        num_epochs,
        config,
        log_dir=None,
        save_dir=None
):
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
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()

        # Update lr
        for param_group in optimiser.param_groups:
            param_group['lr'] = lrs[epoch].item()
        # Update wd
        for param_group in optimiser.param_groups:
            if param_group['weight_decay'] != 0:
                param_group['weight_decay'] = wds[epoch].item()

        epoch_train_pos_loss = []
        epoch_train_neg_loss = []
        epoch_train_loss = []
        epoch_train_pos_acc = []
        epoch_train_neg_acc = []
        epoch_train_acc = []
        epoch_val_pos_loss = []
        epoch_val_neg_loss = []
        epoch_val_loss = []
        epoch_val_pos_acc = []
        epoch_val_neg_acc = []
        epoch_val_acc = []
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        for one, two, three in loop:

            with torch.cuda.amp.autocast():
                # Forward pass
                emb1, emb2, emb3 = model(one), model(two), model(three)
                classify1 = model.classify((emb1 - emb2).square())
                classify2 = model.classify((emb1 - emb3).square())

                targets1 = torch.zeros_like(classify1)
                targets1[:, 0] = 1.0
                targets2 = torch.zeros_like(classify2)
                targets2[:, 1] = 1.0
                
                pos_loss = F.cross_entropy(classify1, targets1)
                neg_loss = F.cross_entropy(classify2, targets2)
                loss = pos_loss + neg_loss

                pos_acc = (classify1.argmax(1) == 0).float().mean()
                neg_acc = (classify2.argmax(1) == 1).float().mean()
                acc = (pos_acc + neg_acc) / 2


            # Backward pass and optimization
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            epoch_train_pos_loss.append(pos_loss.item())
            epoch_train_neg_loss.append(neg_loss.item())
            epoch_train_loss.append(loss.item())
            epoch_train_pos_acc.append(pos_acc.item())
            epoch_train_neg_acc.append(neg_acc.item())
            epoch_train_acc.append(acc.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            for one, two, three in val_loader:
                with torch.cuda.amp.autocast():
                    emb1, emb2, emb3 = model(one), model(two), model(three)
                    classify1 = model.classify((emb1 - emb2).square())
                    classify2 = model.classify((emb1 - emb3).square())

                    targets1 = torch.zeros_like(classify1)
                    targets1[:, 0] = 1.0
                    targets2 = torch.zeros_like(classify2)
                    targets2[:, 1] = 1.0

                    pos_loss = F.cross_entropy(classify1, targets1)
                    neg_loss = F.cross_entropy(classify2, targets2)
                    loss = pos_loss + neg_loss

                    pos_acc = (classify1.argmax(1) == 0).float().mean()
                    neg_acc = (classify2.argmax(1) == 1).float().mean()
                    acc = (pos_acc + neg_acc) / 2

                epoch_val_pos_loss.append(pos_loss.item())
                epoch_val_neg_loss.append(neg_loss.item())
                epoch_val_loss.append(loss.item())
                epoch_val_pos_acc.append(pos_acc.item())
                epoch_val_neg_acc.append(neg_acc.item())
                epoch_val_acc.append(acc.item())
            val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_dir is not None:
                    torch.save(model.state_dict(), save_dir)
        
        if log_dir is not None:
            writer.add_scalar('Loss/train_pos', sum(epoch_train_pos_loss) / len(epoch_train_pos_loss), epoch)
            writer.add_scalar('Loss/train_neg', sum(epoch_train_neg_loss) / len(epoch_train_neg_loss), epoch)
            writer.add_scalar('Loss/train', sum(epoch_train_loss) / len(epoch_train_loss), epoch)
            writer.add_scalar('Accuracy/train', sum(epoch_train_acc) / len(epoch_train_acc), epoch)
            writer.add_scalar('Accuracy/train_pos', sum(epoch_train_pos_acc) / len(epoch_train_pos_acc), epoch)
            writer.add_scalar('Accuracy/train_neg', sum(epoch_train_neg_acc) / len(epoch_train_neg_acc), epoch)

            writer.add_scalar('Loss/val_pos', sum(epoch_val_pos_loss) / len(epoch_val_pos_loss), epoch)
            writer.add_scalar('Loss/val_neg', sum(epoch_val_neg_loss) / len(epoch_val_neg_loss), epoch)
            writer.add_scalar('Loss/val', sum(epoch_val_loss) / len(epoch_val_loss), epoch)
            writer.add_scalar('Accuracy/val', sum(epoch_val_acc) / len(epoch_val_acc), epoch)
            writer.add_scalar('Accuracy/val_pos', sum(epoch_val_pos_acc) / len(epoch_val_pos_acc), epoch)
            writer.add_scalar('Accuracy/val_neg', sum(epoch_val_neg_acc) / len(epoch_val_neg_acc), epoch)
