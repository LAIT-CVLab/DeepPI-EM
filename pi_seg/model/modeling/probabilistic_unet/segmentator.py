import torch.nn as nn
import torch

import numpy as np

from torch.distributions import kl

from pi_seg.model.modeling.probabilistic_unet.backbone import *


device = 'cpu'
try:
    if torch.backends.mps.is_available():
        device = 'mps'
except:
    if torch.cuda.is_available():
        device = 'cuda'


class ProbabilisticUnet(nn.Module):
    def __init__(self, input_channels=64+1, num_classes=1, num_filters=[64, 128, 256, 512, 1024], latent_dim=6, no_convs_fcomb=2, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers,).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)

        self.unet = None
        
    def forward(self, patch, segm=None, coord_features=None, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if coord_features is not None:
            patch = torch.cat([patch, coord_features], 1)
        
        if training and segm is not None:
            self.posterior_latent_space = self.posterior.forward(patch, segm)

        self.prior_latent_space = self.prior.forward(patch)
        self.unet_features = self.unet.forward(patch)
        
    def sample(self, testing=False):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if testing == False:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            #You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior
        
        x = self.fcomb.forward(self.unet_features, z_prior)
        
        return x


    def reconstruct(self, use_posterior_mean=False, calculate_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()

        x = self.fcomb.forward(self.unet_features, z_posterior)

        return x

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
            
        return kl_div

    def elbo(self, segm, ps=None, un_weight=False, un_scale=(2, 1), analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        uncertainty_weight = None
        if un_weight:
            ps = [torch.sigmoid(s) for s in ps]

            uncertainty_map = torch.stack(ps).std(dim=0, unbiased=False)
            uncertainty_map = np.array(uncertainty_map.detach().cpu())
            uncertainty_weight = np.array([np.interp(u, (u.min(), u.max()), (0-un_scale[0], un_scale[1])) for u in uncertainty_map])
            uncertainty_weight = torch.tensor(uncertainty_weight)
            uncertainty_weight = torch.abs(uncertainty_weight).cuda()

        criterion = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=uncertainty_weight)
        z_posterior = self.posterior_latent_space.rsample()

        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False, z_posterior=z_posterior))

        #Here we use the posterior sample sampled above
        self.reconstruction = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calculate_posterior=False, z_posterior=z_posterior)
        self.reconstruction_loss = criterion(input=self.reconstruction, target=segm)

        loss = (self.reconstruction_loss + self.beta * self.kl)
        
        return {'loss': loss, 'instances': self.reconstruction}