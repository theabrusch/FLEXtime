import torch
import torch.nn.functional as F

def mask_generator(
    batch_size: int,
    shape: tuple,
    device: str,
    num_cells: int = 50,
    probablity_of_drop: float = 0.5,
    num_spatial_dims: int = 1,
    dtype = torch.float32, 
    interpolation = 'nearest',
    time_dim = 1):
    """
    Generates a batch of masks by sampling Bernoulli random variables (probablity_of_drop) in a lower dimensional grid (num_cells)
    and upsamples the discrete masks using bilinear interpolation to obtain smooth continious mask in (0, 1).
    """
    length = shape[time_dim]
    if time_dim == 1:
        dim = shape[2]
    else:
        if len(shape) == 2:
            dim = shape[0]
        else:
            dim = shape[1]
    pad_size = (num_cells // 2, num_cells // 2) * num_spatial_dims
    
    if num_spatial_dims == 1:
        grid = (torch.rand(batch_size, dim, *((num_cells,) * num_spatial_dims)) < probablity_of_drop).float()
        if interpolation == 'nearest':
            grid_up = F.interpolate(grid, size=num_cells*5, mode='nearest')
        grid_up = F.interpolate(grid, size=length, mode='linear', align_corners=False)
    else:
        grid = (torch.rand(batch_size, *shape, *((num_cells,) * num_spatial_dims)) < probablity_of_drop).float()
        grid_up = F.interpolate(grid, size=shape, mode='bilinear', align_corners=False)
    
    if device == 'mps':
        grid_up = F.pad(grid_up, pad_size, mode='reflect')
        shift_x = torch.randint(0, num_cells, (batch_size,))
        shift_y = torch.randint(0, num_cells, (batch_size,))
        masks = torch.empty((batch_size, *shape), dtype = dtype)
    else:
        grid_up = F.pad(grid_up, pad_size, mode='reflect').to(device)
        shift_x = torch.randint(0, num_cells, (batch_size,), device=device)
        shift_y = torch.randint(0, num_cells, (batch_size,), device=device)
        masks = torch.empty((batch_size, *shape[1:]), device=device, dtype = dtype)

    for mask_i in range(batch_size):
        if num_spatial_dims == 1:
            m = grid_up[
                mask_i,
                :,
                shift_x[mask_i]:shift_x[mask_i] + length
                ]
            if time_dim == 1:
                m = m.permute(1, 0)
            masks[mask_i] = m
        else:
            masks[mask_i] = grid_up[
                mask_i,
                :,
                shift_y[mask_i]:shift_y[mask_i] + shape[-2],
                shift_x[mask_i]:shift_x[mask_i] + shape[-1]
                ]
        
    yield masks

