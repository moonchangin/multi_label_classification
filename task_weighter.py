import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
from lib.utils import Momentum_Average, Variance_Estimator, cal_grad_norm

class Task_Weighter(nn.Module):
    """
    A module that computes the weight for each task in a multi-label classification problem.

    Args:
        task_num (int): The number of tasks.

    Attributes:
        task_num (int): The number of tasks.
        alpha (torch.Tensor): The weight parameters for each task.
        average_aux_loss (Momentum_Average): A momentum-based average for auxiliary losses.
        gradient_variance_estimator (Variance_Estimator): An estimator for gradient variance.

    Methods:
        forward(losses): Computes the weighted loss based on the given losses.
        gradient_variance_update(model): Updates the gradient variance estimator.
        inject_grad_noise(model, noise_var, lr, momentum): Injects noise into gradients.
        weight_loss(model, losses, reweight_alg): Computes the weighted loss based on the given algorithm.
        arml_weight_loss(model, losses): Computes the ARML (Auxiliary Task Reweighting for Minimum-data Learning) weight loss.
        ada_loss_forward(model, losses): Computes the AdaLoss weight loss.
        gradnorm_weight_loss(model, losses): Computes the GradNorm weight loss.
        cosine_similarity_forward(model, losses, if_update_weight): Computes the weight loss using cosine similarity.
        ol_aux_weight_loss(model, losses): Computes the OL-Aux (Online Auxiliary) weight loss.
    """

    def __init__(self, task_num):
        super(Task_Weighter, self).__init__()
        self.task_num = task_num # Number of tasks
        self.alpha = torch.ones(task_num - 1, requires_grad=True) # Weight parameters for each task
        self.alpha = nn.Parameter(self.alpha) # Convert to a parameter

        self.average_aux_loss = Momentum_Average() # Momentum-based average for auxiliary losses
        self.gradient_variance_estimator = Variance_Estimator() # Estimator for gradient variance

    def forward(self, losses): # Compute the weighted loss based on the given losses
        main_loss = losses[0] # Main loss
        aux_loss = torch.stack(losses[1:], dim=0) # Auxiliary losses
        return main_loss + (aux_loss * self.alpha).sum() # Weighted loss

    def gradient_variance_update(self, model): # Update the gradient variance estimator
        for name, param in model.named_parameters(): # Iterate through the model parameters
            self.gradient_variance_estimator.update(param.grad.data) # Update the gradient variance estimator
            break

    def inject_grad_noise(self, model, noise_var, lr, momentum): # Inject noise into gradients
        for name, param in model.named_parameters(): # Iterate through the model parameters
            if param.grad is not None: #    If the gradient is not None
                param.grad.data = param.grad.data + \
                                  torch.randn_like(param.grad.data) * noise_var**0.5 * (1 - momentum)**0.5 / lr * (param.grad.data**2)**0.25 # Inject noise into the gradient

    def weight_loss(self, model, losses, reweight_alg='arml'): # Compute the weighted loss based on the given algorithm
        weight_loss = None
        if reweight_alg == 'arml': #default
            weight_loss = self.arml_weight_loss(model, losses) # ARML
        elif reweight_alg == 'gradnorm':
            weight_loss = self.gradnorm_weight_loss(model, losses)  # gradnorm
        elif reweight_alg == 'ol_aux':
            weight_loss = self.ol_aux_weight_loss(model, losses)  # ol_aux
        return weight_loss

    def arml_weight_loss(self, model, losses): # Compute the ARML weight loss
        main_loss = losses[0] # Main loss
        aux_loss = torch.stack(losses[1:], dim=0) # Auxiliary losses
        # Compute the gradient gap between the main loss and the auxiliary loss
        loss_grad_gap = grad(main_loss - (aux_loss * self.alpha).sum(), model.parameters(), 
                          create_graph=True, allow_unused=True)
        # Compute the alpha loss
        alpha_loss = sum([grd.norm()**2 for grd in loss_grad_gap[:-5]]) # default is 6; 5 is the last layer
        return alpha_loss

    # def ada_loss_forward(self, model, losses): # Compute the AdaLoss weight loss
    #     main_loss = losses[0]
    #     losses[2] = losses[2] + 1
    #     aux_loss = torch.log(torch.stack(losses[1:], dim=0) + 1e-6)
    #     return main_loss + (aux_loss * self.alpha).sum()
    def ada_loss_forward(self, model, losses):
        losses_list = list(losses)# Convert the tuple of losses to a list for mutability
        
        # Perform the original operations using the list
        main_loss = losses_list[0]
        losses_list[2] = losses_list[2] + 1  # This is now possible
        aux_loss = torch.log(torch.stack(losses_list[1:], dim=0) + 1e-6)
        
        # Return the sum of the main loss and weighted auxiliary losses
        return main_loss + (aux_loss * self.alpha).sum()


    def gradnorm_weight_loss(self, model, losses):
        main_loss = losses[0]
        aux_loss = torch.stack(losses[1:], dim=0)
        self.average_aux_loss.update(aux_loss.detach())
        r = aux_loss.detach() / self.average_aux_loss.init
        r = r / r.mean()

        alpha_loss = 0
        grad_norms = []
        for i in range(1, self.task_num):
            grad_i = grad(aux_loss[i - 1], model.layer2.parameters(),
                            create_graph=True, allow_unused=True)
            grad_norms.append(self.alpha[i - 1] * grad_i[-1].detach().norm())

        grad_norm_mean = sum(grad_norms) / (self.task_num - 1)
        for i in range(1, self.task_num):
            alpha_loss += torch.abs(grad_norms[i - 1] - grad_norm_mean * r[i - 1]**1.0)

        return alpha_loss

    def cosine_similarity_forward(self, model, losses, if_update_weight):
        losses = list(losses)# Convert the tuple of losses to a list for mutability
        main_loss = losses[0]
        aux_loss = torch.stack(losses[1:], dim=0)

        if if_update_weight:
            main_grad = grad(main_loss, model.layer2.parameters(), retain_graph=True)[-1].detach() # 'MLCModel' object has no attribute 'unit3'
            for i in range(1, self.task_num):
                grad_i = grad(aux_loss[i - 1], model.layer2.parameters(), retain_graph=(i < self.task_num - 1))[-1].detach()
                self.alpha.data[i - 1] = int((grad_i * main_grad).sum() > 0)

        return main_loss + (aux_loss * self.alpha).sum()

    def ol_aux_weight_loss(self, model, losses):
        main_loss = losses[0]
        aux_loss = torch.stack(losses[1:], dim=0)

        main_loss_grads = grad(main_loss, model.parameters(),
                          create_graph=True, allow_unused=True)
        aux_loss_grads = grad((aux_loss * self.alpha).sum(), model.parameters(),
                          create_graph=True, allow_unused=True)

        alpha_loss = 0
        for main_grad, aux_grad in zip(main_loss_grads[:-5], aux_loss_grads[:-5]): #default is 6; 5 is the last layer
            alpha_loss += -(main_grad.detach() * aux_grad).sum()
        return alpha_loss



def task_weighter(task_num):
    return Task_Weighter(task_num)