import torch
import numpy as np

class FadeMovingAverageWindow():
    """This class allows to create and apply 'fade to moving average' perturbations on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor): The mask tensor than indicates the intensity of the perturbation
            to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
        window_size: Size of the window where each moving average is computed (called W in the paper).
    """

    def __init__(self, device, window_size=2, eps=1.0e-7):
        self.eps = eps
        self.device = device
        self.window_size = window_size

    def apply(self, X, mask_tensor, time_dim = -1):
        T = X.shape[time_dim]
        X = X.squeeze(0).float()
        unsqueeze = False
        squeeze = False
        mask_tensor = mask_tensor.squeeze(0)
        if time_dim == -1:
            X = X.transpose(-1, -2)
            mask_tensor = mask_tensor.transpose(-1, -2)
        if len(X.shape) > 2:
            unsqueeze = True
            X = X.squeeze(0)
            mask_tensor = mask_tensor.squeeze(0)
        elif len(X.shape) == 1:
            X = X.unsqueeze(1)
            squeeze = True
            mask_tensor = mask_tensor.unsqueeze(1)
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # For each feature and each time, we compute the coefficients of the perturbation tensor
        T1_tensor = T_axis.unsqueeze(1)
        T2_tensor = T_axis.unsqueeze(0)
        filter_coefs = torch.abs(T1_tensor - T2_tensor) <= self.window_size
        filter_coefs = filter_coefs / (2 * self.window_size + 1)
        X_avg = torch.einsum("st,si->ti", filter_coefs.float(), X)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = X_avg + mask_tensor * (X - X_avg)
        X_pert = X_pert.unsqueeze(0)
        if time_dim == -1:
            X_pert = X_pert.transpose(-1, -2)
        if unsqueeze:
            X_pert = X_pert.unsqueeze(0)
        elif squeeze:
            X_pert = X_pert.squeeze()
        return X_pert

    def apply_extremal(self, X: torch.Tensor, masks_tensor: torch.Tensor):
        N_area, T, N_features = masks_tensor.shape
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # For each feature and each time, we compute the coefficients for the Gaussian perturbation
        T1_tensor = T_axis.unsqueeze(1)
        T2_tensor = T_axis.unsqueeze(0)
        filter_coefs = torch.abs(T1_tensor - T2_tensor) <= self.window_size
        filter_coefs = filter_coefs / (2 * self.window_size + 1)
        X_avg = torch.einsum("st,si->ti", filter_coefs, X[0, :, :])
        X_avg = X_avg.unsqueeze(0)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = X_avg + masks_tensor * (X - X_avg)
        return X_pert



class FadeMovingAverage():
    """This class allows to create and apply 'fade to moving average' perturbations on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor): The mask tensor than indicates the intensity of the perturbation
            to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
    """

    def __init__(self, device, eps=1.0e-7):
        self.eps = eps
        self.device = device

    def apply(self, X, mask_tensor, time_dim = -1):
        T = X.shape[time_dim]
        X = X.squeeze(0)
        unsqueeze = False
        if len(X.shape) > 2:
            unsqueeze = True
            X = X.squeeze(0)
            mask_tensor = mask_tensor.squeeze(0)
            time_dim = 2
        mask_tensor = mask_tensor.squeeze(0)
        # Compute the moving average for each feature and concatenate it to create a tensor with X's shape
        moving_average = torch.mean(X, (time_dim-1).item()).reshape(1, -1).to(self.device)
        moving_average_tiled = moving_average.repeat(T, 1)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = mask_tensor * X + (1 - mask_tensor) * moving_average_tiled
        if unsqueeze:
            X_pert = X_pert.unsqueeze(0)
        return X_pert.unsqueeze(0)

    def apply_extremal(self, X, extremal_tensor: torch.Tensor):
        # Compute the moving average for each feature and concatenate it to create a tensor with X's shape
        moving_average = torch.mean(X, dim=0).reshape(1, 1, -1).to(self.device)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = extremal_tensor * X + (1 - extremal_tensor) * moving_average
        return X_pert

class GaussianBlur():
    """This class allows to create and apply 'Gaussian blur' perturbations on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor): The mask tensor than indicates the intensity of the perturbation
            to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
        sigma_max (float): Maximal width for the Gaussian blur.
    """

    def __init__(self, device, eps=1.0e-7, sigma_max=2):
        self.sigma_max = sigma_max
        self.device = device
        self.eps = eps

    def apply(self, X, mask_tensor, time_dim = -1):
        T = X.shape[time_dim]
        X = X.squeeze(0)
        if len(X.shape) > 2:
            unsqueeze = True
            X = X.squeeze(0)
            mask_tensor = mask_tensor.squeeze(0)
        mask_tensor = mask_tensor.squeeze(0)
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # Convert the mask into a tensor containing the width of each Gaussian perturbation
        sigma_tensor = self.sigma_max * ((1 + self.eps) - mask_tensor)
        sigma_tensor = sigma_tensor.unsqueeze(0)
        # For each feature and each time, we compute the coefficients for the Gaussian perturbation
        T1_tensor = T_axis.unsqueeze(1).unsqueeze(2)
        T2_tensor = T_axis.unsqueeze(0).unsqueeze(2)
        filter_coefs = torch.exp(torch.divide(-1.0 * (T1_tensor - T2_tensor) ** 2, 2.0 * (sigma_tensor ** 2)))
        filter_coefs = torch.divide(filter_coefs, torch.sum(filter_coefs, 0))
        # The perturbation is obtained by replacing each input by the linear combination weighted by Gaussian coefs
        X_pert = torch.einsum("sti,si->ti", filter_coefs, X)
        if unsqueeze:
            X_pert = X_pert.unsqueeze(0)
        return X_pert.unsqueeze(0)

    def apply_extremal(self, X: torch.Tensor, extremal_tensor: torch.Tensor):
        N_area, T, N_features = extremal_tensor.shape
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # Convert the mask into a tensor containing the width of each Gaussian perturbation
        sigma_tensor = self.sigma_max * ((1 + self.eps) - extremal_tensor).reshape(N_area, 1, T, N_features)
        # For each feature and each time, we compute the coefficients for the Gaussian perturbation
        T1_tensor = T_axis.reshape(1, 1, T, 1)
        T2_tensor = T_axis.reshape(1, T, 1, 1)
        filter_coefs = torch.exp(torch.divide(-1.0 * (T1_tensor - T2_tensor) ** 2, 2.0 * (sigma_tensor ** 2)))
        filter_coefs = filter_coefs / torch.sum(filter_coefs, dim=1, keepdim=True)
        # The perturbation is obtained by replacing each input by the linear combination weighted by Gaussian coefs
        X_pert = torch.einsum("asti,si->ati", filter_coefs, X)
        return X_pert
    

class FreqMask():
    def __init__(self, model, regularization = 'l1', device = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.regularization = regularization

    def fit(self, 
            data,
            additional_inputs = None,
            time_dim = -1,
            n_epoch: int = 500,
            learning_rate: float = 1.0e-1,
            momentum: float = 0.9,
            logger = None,
            keep_ratio: float = 0.05,
            reg_factor_init: float = 1.,
            time_reg_strength: float = 0.,
            reg_factor_dilation: float = 1.,
            verbose = True,
            stop_criterion = 1.0e-6,
            use_only_max_target = True,
            patience = 10,
            add_noise = False,
            perturb = 'fade',
            return_epochs = False
            ):
        shape = data.shape
        self.model.eval()
        # initialize mask
        with torch.no_grad():
            target = self.model(data.float().to(self.device), additional_inputs)
            # softmax
            target = torch.nn.functional.softmax(target, dim = 1)
            # set all other than maximum target to 0
            if use_only_max_target:
                target = torch.argmax(target, dim = 1)
                # target_temp = torch.zeros_like(target)
                # target_temp[torch.arange(target.shape[0]), max_target] = target[:, max_target]
                # target = target_temp

        fft_data = torch.fft.rfft(data, dim = time_dim).to(self.device)
        fft_imag = fft_data.imag
        fft_data = fft_data.real
        mask = 0.5*torch.ones_like(fft_data).to(self.device)
        mask.requires_grad = True
        time_dim = torch.tensor(time_dim).to(self.device)

        if self.regularization == 'ratio':
            reg_ref = torch.zeros(int((1 - keep_ratio) * mask.shape[-1]))
            reg_ref = torch.cat((reg_ref, torch.ones(mask.shape[-1] - reg_ref.shape[0]))).to(self.device)
        reg_strength = reg_factor_init
    
        reg_multiplicator = np.exp(np.log(reg_factor_dilation) / n_epoch)

        optimizer = torch.optim.SGD([mask], lr = learning_rate, momentum = momentum)
        l = float('inf')
        total_loss = []
        early_stopping_counter = 0
        if perturb == 'window':
            perturb = FadeMovingAverageWindow(device = self.device, window_size = 10)
        elif perturb == 'blur':
            perturb = GaussianBlur(device = self.device)
        elif perturb == 'fade':
            perturb = FadeMovingAverage(device = self.device)
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            if add_noise:
                input_fft = fft_data * mask + (1-mask) * torch.randn_like(fft_data)*fft_data.std()*add_noise
            elif perturb:
                input_fft = fft_data * mask + (1-mask) * perturb.apply(fft_data, mask, time_dim=time_dim)
            else:
                input_fft = fft_data * mask
            input_data = torch.fft.irfft(input_fft + 1j*fft_imag, dim = time_dim)
            output = self.model(input_data.float(), additional_inputs)
            output = torch.nn.functional.softmax(output, dim = 1)
            target_loss = self.loss_function(output, target)
            if self.regularization == 'ratio':
                mask_tensor_sorted = torch.sort(mask)[0]
                reg_loss = ((reg_ref - mask_tensor_sorted)**2).mean()
            elif self.regularization == 'l1':
                reg_loss = torch.max(mask.abs().mean()-torch.tensor(keep_ratio).to(self.device), torch.tensor(0.).to(self.device))
            time_reg = (torch.abs(torch.index_select(mask, time_dim, torch.arange(1, mask.shape[time_dim]-1).to(self.device)) - torch.index_select(mask, time_dim, torch.arange(0, mask.shape[time_dim]-2).to(self.device)))).mean()
            if epoch < 10:
                time_strength = 0
            else:
                time_strength = time_reg_strength
            if time_strength < 1e-6:
                loss = target_loss + reg_strength * reg_loss
            else:
                loss = target_loss + reg_strength * reg_loss + time_strength * time_reg
            loss.backward()
            optimizer.step()
            # make sure mask is between 0 and 1
            mask.data = torch.clamp(mask, 0, 1)
            total_loss.append(loss.item())
            reg_strength *= reg_multiplicator
            if verbose:
                print(f'Epoch: {epoch} Loss: {loss.item()}, Target loss: {target_loss.item()}, Reg loss: {reg_loss.item()}')
            if logger is not None:
                logger.log({'loss': loss.item()})
            
            if stop_criterion is not None:
                if abs(l - loss.item()) < stop_criterion:
                    early_stopping_counter += 1
                l = loss.item()
                if early_stopping_counter > patience:
                    break
        # clear gpu memory
        mask = mask.cpu().detach()
        del input_fft, input_data, output, target, fft_data, fft_imag
        torch.cuda.empty_cache()
        if return_epochs:
            return mask, epoch
        return mask, total_loss
    
class CaptumFreqMask():
    def __init__(self, model, target, optimization_params, additional_input = None, regularization = 'l1', device = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.regularization = regularization
        self.additional_input = additional_input 
        self.optimization_params = optimization_params
        self.target = target

    def fit(self, 
            data,
            additional_inputs = None,
            time_dim = -1,
            n_epoch: int = 500,
            learning_rate: float = 1.0e-1,
            momentum: float = 0.9,
            logger = None,
            keep_ratio: float = 0.05,
            reg_factor_init: float = 1.,
            time_reg_strength: float = 0.,
            reg_factor_dilation: float = 1.,
            verbose = True,
            stop_criterion = 1.0e-6,
            use_only_max_target = True,
            patience = 10,
            perturb = False,
            add_noise = False,
            ):
        shape = data.shape
        self.model.eval()
        # initialize mask
        with torch.no_grad():
            target = self.model(data.float().to(self.device), additional_inputs)
            # softmax
            target = torch.nn.functional.softmax(target, dim = 1)
            # set all other than maximum target to 0
            if use_only_max_target:
                target = torch.argmax(target, dim = 1)
                # target_temp = torch.zeros_like(target)
                # target_temp[torch.arange(target.shape[0]), max_target] = target[:, max_target]
                # target = target_temp

        fft_data = torch.fft.rfft(data, dim = time_dim).to(self.device)
        fft_imag = fft_data.imag
        fft_data = fft_data.real
        mask = 0.5*torch.ones_like(fft_data).to(self.device)
        mask.requires_grad = True
        time_dim = torch.tensor(time_dim).to(self.device)

        if self.regularization == 'ratio':
            reg_ref = torch.zeros(int((1 - keep_ratio) * mask.shape[-1]))
            reg_ref = torch.cat((reg_ref, torch.ones(mask.shape[-1] - reg_ref.shape[0]))).to(self.device)
        reg_strength = reg_factor_init
    
        reg_multiplicator = np.exp(np.log(reg_factor_dilation) / n_epoch)

        optimizer = torch.optim.SGD([mask], lr = learning_rate, momentum = momentum)
        l = float('inf')
        total_loss = []
        early_stopping_counter = 0
        if perturb == 'window':
            perturb = FadeMovingAverageWindow(device = self.device, window_size = 10)
        elif perturb == 'blur':
            perturb = GaussianBlur(device = self.device)
        elif perturb == 'fade':
            perturb = FadeMovingAverage(device = self.device)
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            if add_noise:
                input_fft = fft_data * mask + (1-mask) * torch.randn_like(fft_data)*fft_data.std()*add_noise
            elif perturb:
                input_fft = fft_data * mask + (1-mask) * perturb.apply(fft_data, mask, time_dim=time_dim)
            else:
                input_fft = fft_data * mask
            input_data = torch.fft.irfft(input_fft + 1j*fft_imag, dim = time_dim)
            output = self.model(input_data.float(), additional_inputs)
            output = torch.nn.functional.softmax(output, dim = 1)
            target_loss = self.loss_function(output, target)
            if self.regularization == 'ratio':
                mask_tensor_sorted = torch.sort(mask)[0]
                reg_loss = ((reg_ref - mask_tensor_sorted)**2).mean()
            elif self.regularization == 'l1':
                reg_loss = torch.max(mask.abs().mean()-torch.tensor(keep_ratio).to(self.device), torch.tensor(0.).to(self.device))
            time_reg = (torch.abs(torch.index_select(mask, time_dim, torch.arange(1, mask.shape[time_dim]-1).to(self.device)) - torch.index_select(mask, time_dim, torch.arange(0, mask.shape[time_dim]-2).to(self.device)))).mean()
            if epoch < 10:
                time_strength = 0
            else:
                time_strength = time_reg_strength
            if time_strength < 1e-6:
                loss = target_loss + reg_strength * reg_loss
            else:
                loss = target_loss + reg_strength * reg_loss + time_strength * time_reg
            loss.backward()
            optimizer.step()
            # make sure mask is between 0 and 1
            mask.data = torch.clamp(mask, 0, 1)
            total_loss.append(loss.item())
            reg_strength *= reg_multiplicator
            if verbose:
                print(f'Epoch: {epoch} Loss: {loss.item()}, Target loss: {target_loss.item()}, Reg loss: {reg_loss.item()}')
            if logger is not None:
                logger.log({'loss': loss.item()})
            
            if stop_criterion is not None:
                if abs(l - loss.item()) < stop_criterion:
                    early_stopping_counter += 1
                l = loss.item()
                if early_stopping_counter > patience:
                    break
        # clear gpu memory
        mask = mask.cpu().detach()
        del input_fft, input_data, output, target, fft_data, fft_imag
        torch.cuda.empty_cache()

        return mask, total_loss
    
    def __call__(self, inputs, device = 'cpu', **kwargs):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        masks = []
        for x in inputs:
            with torch.enable_grad():
                mask, _ = self.fit(x.unsqueeze(0), additional_inputs = self.additional_input, **self.optimization_params, verbose = False)
            mask = mask.cpu().detach().unsqueeze(0)
            masks.append(mask)
        masks = torch.stack(masks)
        if self.optimization_params['time_dim'] == -1:
            # reshape all other dimensions than time to fit input
            masks.squeeze()
            masks = masks.reshape(*inputs.shape[:-1], masks.shape[-1])
        else:
            masks = masks.reshape(*inputs.shape[:-2], masks.shape[-2], inputs.shape[-1])
        return masks