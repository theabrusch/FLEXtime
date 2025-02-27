import torch
import numpy as np

class FLEXtime():
    def __init__(self, model, filterbank, regularization = 'l1', device = 'cpu'):
        self.model = model.to(device)
        self.filterbank = filterbank
        self.device = device
        self.nfilters = filterbank.numbanks
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.regularization = regularization

    def fit(self, 
            data,
            additional_input = None,
            time_dim = -1,
            n_epoch: int = 500,
            learning_rate: float = 1.0e-1,
            momentum: float = 0.9,
            logger = None,
            keep_ratio: float = 0.05,
            reg_factor_init: float = 1,
            reg_factor_dilation: float = 1,
            stop_criterion: float = 1.0e-6,
            time_reg_strength = 0,
            patience: int = 10,
            verbose: bool = True,
            use_only_max_target: bool = True,
            add_noise: bool = False,
            return_epochs =False,
            ):
        shape = data.shape
        self.model.eval()
        
        with torch.no_grad():
            target = self.model(data.float().to(self.device), additional_input)
            # softmax
            target = torch.nn.functional.softmax(target, dim = 1)
            # set all other than maximum target to 0
            if use_only_max_target:
                target = torch.argmax(target, dim = 1)
                #target_temp = torch.zeros_like(target)
                #target_temp[torch.arange(target.shape[0]), max_target] = target[:, max_target]
                #target = target_temp

        # initialize mask
        mask_shape = torch.tensor(shape)
        mask_shape[time_dim] = 1
        mask = 0.5*torch.ones((*mask_shape, self.nfilters), device = self.device)
        mask.requires_grad = True
        optimizer = torch.optim.SGD([mask], lr = learning_rate, momentum = momentum)
        # initialize regularization reference
        if self.regularization == 'ratio':
            reg_ref = torch.zeros(int((1 - keep_ratio) * self.nfilters))
            reg_ref = torch.cat((reg_ref, torch.ones(self.nfilters - reg_ref.shape[0]))).to(self.device)
        
        reg_strength = reg_factor_init
        reg_multiplicator = np.exp(np.log(reg_factor_dilation) / n_epoch)
        # compute filterbank
        bands = self.filterbank.apply_filter_bank(data.cpu().numpy(), adjust_for_delay = True, time_dim = time_dim)
        bands = torch.tensor(bands).float().to(self.device).reshape(*shape, self.nfilters)
        time_dim = torch.tensor(time_dim).to(self.device)
        l = float('inf')
        total_loss = []
        early_stopping_counter = 0
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            if add_noise:
                input_data = (bands * mask + (1-mask) * torch.randn_like(bands)*bands.std()*add_noise).sum(-1)
            else:
                input_data = (bands * mask).sum(-1)
            output = self.model(input_data, additional_input)
            output = torch.nn.functional.softmax(output, dim = 1)
            target_loss = self.loss_function(output, target)
            if self.regularization == 'ratio':
                mask_tensor_sorted = torch.sort(mask)[0]
                reg_loss = ((reg_ref - mask_tensor_sorted)**2).mean()
            elif self.regularization == 'l1':
                reg_loss = torch.max(mask.abs().mean()-torch.tensor(keep_ratio).to(self.device), torch.tensor(0.).to(self.device))
            if epoch < 10:
                time_strength = 0
            else:
                time_strength = time_reg_strength
            if time_strength < 1e-6:
                loss = target_loss + reg_strength * reg_loss
            else:
                time_reg = (torch.abs(torch.index_select(mask, -1, torch.arange(1, mask.shape[-1]-1).to(self.device)) - torch.index_select(mask, -1, torch.arange(0, mask.shape[-1]-2).to(self.device)))).mean()
                loss = target_loss + reg_strength * reg_loss + time_strength*time_reg
            loss.backward()
            optimizer.step()
            # make sure mask is between 0 and 1
            mask.data = torch.clamp(mask, 0, 1)
            total_loss.append(loss.item())
            reg_strength *= reg_multiplicator
            
            if verbose:
                print(f'Epoch: {epoch} Loss: {loss.item()}, Target loss: {target_loss.item()}, Reg loss: {reg_loss.item()}, L1: {mask.abs().mean()}')
            if logger is not None:
                logger.log({'loss': loss.item()})

            if stop_criterion is not None:
                if abs(l - loss.item()) < stop_criterion:
                    early_stopping_counter += 1
                l = loss.item()
                if early_stopping_counter > patience:
                    break
        if return_epochs:
            return mask, epoch
        return mask, total_loss

class CaptumFLEXtime():
    '''
    Version compatible with Captum for computing sensitivity scoress
    '''
    def __init__(self, model, target, filterbank, optimization_params, additional_input = None, regularization = 'l1', device = 'cpu'):
        self.model = model.to(device)
        self.filterbank = filterbank
        self.device = device
        self.nfilters = filterbank.numbanks
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.regularization = regularization
        self.additional_input = additional_input 
        self.optimization_params = optimization_params
        self.target = target

    def fit(self, 
            data,
            additional_input = None,
            time_dim = -1,
            n_epoch: int = 500,
            learning_rate: float = 1.0e-1,
            momentum: float = 0.9,
            logger = None,
            keep_ratio: float = 0.05,
            reg_factor_init: float = 1,
            reg_factor_dilation: float = 1,
            stop_criterion: float = 1.0e-6,
            patience = 10,
            verbose: bool = True,
            time_reg_strength = 0,
            ):
        shape = data.shape
        self.model.eval()
        
        with torch.no_grad():
            target = self.model(data.float().to(self.device), additional_input)
            # softmax
            target = torch.nn.functional.softmax(target, dim = 1)
            # set all other than maximum target to 0
            target = torch.argmax(target, dim = 1)
            #target_temp = torch.zeros_like(target)
            #target_temp[torch.arange(target.shape[0]), max_target] = target[:, max_target]
            #target = target_temp

        # initialize mask
        mask_shape = torch.tensor(shape)
        mask_shape[time_dim] = 1
        mask = 0.5*torch.ones((*mask_shape, self.nfilters), device = self.device)
        mask.requires_grad = True
        optimizer = torch.optim.SGD([mask], lr = learning_rate, momentum = momentum)
        # initialize regularization reference
        if self.regularization == 'ratio':
            reg_ref = torch.zeros(int((1 - keep_ratio) * self.nfilters))
            reg_ref = torch.cat((reg_ref, torch.ones(self.nfilters - reg_ref.shape[0]))).to(self.device)
        
        reg_strength = reg_factor_init
        reg_multiplicator = np.exp(np.log(reg_factor_dilation) / n_epoch)
        # compute filterbank
        bands = self.filterbank.apply_filter_bank(data.cpu().numpy(), adjust_for_delay = True, time_dim = time_dim)
        bands = torch.tensor(bands).float().to(self.device).reshape(*shape, self.nfilters)
        time_dim = torch.tensor(time_dim).to(self.device)
        l = float('inf')
        total_loss = []
        early_stopping_counter = 0
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            input_data = (bands * mask).sum(-1)
            output = self.model(input_data, additional_input)
            output = torch.nn.functional.softmax(output, dim = 1)
            target_loss = self.loss_function(output, target)
            if self.regularization == 'ratio':
                mask_tensor_sorted = torch.sort(mask)[0]
                reg_loss = ((reg_ref - mask_tensor_sorted)**2).mean()
            elif self.regularization == 'l1':
                reg_loss = torch.max(mask.abs().mean()-torch.tensor(keep_ratio).to(self.device), torch.tensor(0.).to(self.device))
            if epoch < 10:
                time_strength = 0
            else:
                time_strength = time_reg_strength
            if time_strength < 1e-6:
                loss = target_loss + reg_strength * reg_loss
            else:
                time_reg = (torch.abs(torch.index_select(mask, -1, torch.arange(1, mask.shape[-1]-1).to(self.device)) - torch.index_select(mask, -1, torch.arange(0, mask.shape[-1]-2).to(self.device)))).mean()
                loss = target_loss + reg_strength * reg_loss + time_strength*time_reg
            loss.backward()
            optimizer.step()
            # make sure mask is between 0 and 1
            mask.data = torch.clamp(mask, 0, 1)
            total_loss.append(loss.item())
            reg_strength *= reg_multiplicator
            
            if verbose:
                print(f'Epoch: {epoch} Loss: {loss.item()}, Target loss: {target_loss.item()}, Reg loss: {reg_loss.item()}, L1: {mask.abs().mean()}')
            if logger is not None:
                logger.log({'loss': loss.item()})

            if stop_criterion is not None:
                if abs(l - loss.item()) < stop_criterion:
                    early_stopping_counter += 1
                l = loss.item()
                if early_stopping_counter > patience:
                    break
        return mask, total_loss

    def __call__(self, inputs, device = 'cpu', **kwargs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        masks = []
        for x in inputs:
            x = x.unsqueeze(0)
            with torch.enable_grad():
                mask, _ = self.fit(x, additional_input = self.additional_input, **self.optimization_params, verbose = False)
                mask = mask.cpu().detach().numpy().squeeze()
            importance = torch.tensor(self.filterbank.get_collect_filter_response(mask)).unsqueeze(0)
            masks.append(importance)
        masks = torch.stack(masks)
        if self.optimization_params['time_dim'] == -1:
            # reshape all other dimensions than time to fit input
            masks = masks.squeeze(-1)
            masks = masks.reshape(*inputs.shape[:-1], masks.shape[-1])
        else:
            masks = masks.reshape(*inputs.shape[:-2], masks.shape[-2], inputs.shape[-1])
        return masks