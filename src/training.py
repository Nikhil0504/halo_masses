import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

seed = 255
torch.manual_seed(seed)

# Initialize the GradScaler outside the function (at the start of training)
# scaler = GradScaler()

def configure_optimizer(model, lr, wd,):
    """Only apply weight decay to weights, but not to other
    parameters like biases or LayerNorm. Based on minGPT version.
    """

    decay, no_decay = set(), set()
    yes_wd_modules = (nn.Linear, )
    no_wd_modules = (nn.LayerNorm, )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, yes_wd_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, no_wd_modules):
                no_decay.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups, 
        lr=lr, 
        # betas=(0.9, 0.999),
    )

    return optimizer

# Training loop
def train(loader, model, optimizer, scheduler, device, augment=True, max_grad_norm=None):
    model.train()

    total_loss = 0

    for data in loader:
        if augment:
            data_node_features_scatter = 3e-4 * torch.randn_like(data.x_hydro) * torch.std(data.x_hydro, dim=0)
            data_edge_features_scatter = 3e-4 * torch.randn_like(data.edge_attr) * torch.std(data.edge_attr, dim=0)
            
            data.x_hydro += data_node_features_scatter
            data.edge_attr += data_edge_features_scatter

            assert not torch.isnan(data.x_hydro).any() 
            assert not torch.isnan(data.edge_attr).any() 

        data.to(device)

        optimizer.zero_grad()

        # with autocast():
        # chunk it into 2 parts
        y_pred, logvar_pred = model(data).chunk(2, dim=1)
        assert not torch.isnan(y_pred).any() and not torch.isnan(logvar_pred).any()
        y_pred = y_pred.view(-1, model.n_out)
        logvar_pred = logvar_pred.mean()
        loss = 0.5 * (F.mse_loss(y_pred, data.y) / 10**logvar_pred + logvar_pred)

        loss.backward()
        # scaler.scale(loss).backward()

        # use gradient clipping
        if max_grad_norm is not None:
            # scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

        total_loss += loss.item()
    
    if scheduler:
        scheduler.step()

    return total_loss / len(loader)


def validate(loader, model, device):
    model.eval()

    total_loss = 0
    total_mse = 0  # For calculating RMSE

    predictions = []
    actuals = []
    subhalo_ids = []
    dists = []
    
    with torch.no_grad():
        for data in loader:
            data.to(device)
            
            # with autocast():
            y_pred, logvar_pred = model(data).chunk(2, dim=1)
            y_pred = y_pred.view(-1, model.n_out)
            logvar_pred = logvar_pred.mean()

            loss = 0.5 * (F.mse_loss(y_pred, data.y) / 10**logvar_pred + logvar_pred)
            mse = F.mse_loss(y_pred, data.y, reduction='sum')  # To calculate RMSE

            total_loss += loss.item()
            total_mse += mse.item()
            predictions += list(y_pred.detach().cpu().numpy())
            actuals += list(data.y.detach().cpu().numpy())
            subhalo_ids += list(data.subhalo_id.detach().cpu().numpy())
            dists += list(data.dist_from_center.detach().cpu().numpy())


    predictions = np.concatenate(predictions)
    actuals = np.array(actuals)
    subhalo_ids = np.array(subhalo_ids)
    rmse = np.sqrt(total_mse / len(actuals))

    return (
        total_loss / len(loader),
        predictions,
        actuals,
        subhalo_ids,
        rmse,  # Optionally return RMSE directly from validation,
        dists
    )