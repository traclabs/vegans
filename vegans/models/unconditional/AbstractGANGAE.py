import torch

from torch.nn import BCELoss, BCEWithLogitsLoss
from vegans.utils import WassersteinLoss
from vegans.utils.networks import Generator, Adversary, Encoder
from vegans.models.unconditional.AbstractGenerativeModel import AbstractGenerativeModel

import numpy as np


class AbstractGANGAE(AbstractGenerativeModel):
    """ Abstract class for GAN with structure of one generator, one discriminator / critic and
    one encoder. Examples are the `LRGAN`, `VAEGAN` and `BicycleGAN`.

    Parameters
    ----------
    generator: nn.Module
        Generator architecture. Produces output in the real space.
    adversary: nn.Module
        Adversary architecture. Produces predictions for real and fake samples to differentiate them.
    encoder : nn.Module
        Encoder architecture. Produces predictions in the latent space.
    x_dim : list, tuple
        Number of the output dimensions of the generator and input dimension of the discriminator / critic.
        In the case of images this will be [nr_channels, nr_height_pixels, nr_width_pixels].
    z_dim : int, list, tuple
        Number of the latent dimensions for the generator input. Might have dimensions of an image.
    optim : dict or torch.optim
        Optimizer used for each network. Could be either an optimizer from torch.optim or a dictionary with network
        name keys and torch.optim as value, i.e. {"Generator": torch.optim.Adam}.
    optim_kwargs : dict
        Optimizer keyword arguments used for each network. Must be a dictionary with network
        name keys and dictionary with keyword arguments as value, i.e. {"Generator": {"lr": 0.0001}}.
    feature_layer : torch.nn.*
        Output layer used to compute the feature loss. Should be from either the discriminator or critic.
        If `feature_layer` is not None, the original generator loss is replaced by a feature loss, introduced
        [here](https://arxiv.org/abs/1606.03498v1).
    fixed_noise_size : int
        Number of images shown when logging. The fixed noise is used to produce the images in the folder/images
        subdirectory, the tensorboard images tab and the samples in get_training_results().
    lambda_grad: float
        Weight for the reconstruction loss of the gradients. Pushes the norm of the gradients to 1.
    device : string
        Device used while training the model. Either "cpu" or "cuda".
    ngpu : int
        Number of gpus used during training if device == "cuda".
    folder : string
        Creates a folder in the current working directory with this name. All relevant files like summary, images, models and
        tensorboard output are written there. Existing folders are never overwritten or deleted. If a folder with the same name
        already exists a time stamp is appended to make it unique.
    """

    #########################################################################
    # Actions before training
    #########################################################################
    def __init__(
            self,
            generator,
            adversary,
            encoder,
            x_dim,
            z_dim,
            optim=None,
            optim_kwargs=None,
            adv_type="Discriminator",
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            folder=None,
            ngpu=0,
            secure=True,
            lr_decay=0.9,
            _called_from_conditional=False):

        self.adv_type = adv_type
        self.generator = Generator(generator, input_size=z_dim, device=device, ngpu=ngpu, secure=secure)
        self.adversary = Adversary(adversary, input_size=x_dim, adv_type=adv_type, device=device, ngpu=ngpu, secure=secure)
        self.encoder = Encoder(encoder, input_size=x_dim, device=device, ngpu=ngpu, secure=secure)
        # self.encoder = encoder
        self.neural_nets = {
            "Encoder": self.encoder, "Generator": self.generator, "Adversary": self.adversary
        }

        super().__init__(
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, ngpu=ngpu, folder=folder, secure=secure, lr_decay=lr_decay
        )
        self.hyperparameters["adv_type"] = adv_type
        if not _called_from_conditional and self.secure:
            assert self.generator.output_size == self.x_dim, (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )

    def _define_loss(self):
        if self.adv_type == "Discriminator":
            loss_functions = {"Generator": BCEWithLogitsLoss(), "Adversary": BCEWithLogitsLoss(), "Encoder": BCEWithLogitsLoss()}
        elif self.adv_type == "Critic":
            loss_functions = {"Generator": WassersteinLoss(), "Adversary": WassersteinLoss()}
        else:
            raise NotImplementedError("'adv_type' must be one of Discriminator or Critic.")
        return loss_functions


    #########################################################################
    # Actions during training
    #########################################################################
    def calculate_losses(self, X_batch, Z_batch, who=None):
        """ Calculates the losses for GANs using a 1v1 architecture.

        This method is called within the `AbstractGenerativeModel` main `fit()` loop.

        Parameters
        ----------
        X_batch : torch.Tensor
            Current x batch.
        Z_batch : torch.Tensor
            Current z batch.
        who : None, optional
            Name of the network that should be trained.
        """
        if who == "Generator":
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Adversary":
            losses = self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch)
        elif who == "Encoder":
            losses = self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch)
        else:
            losses = self._calculate_generator_loss(X_batch=X_batch, Z_batch=Z_batch)
            losses.update(self._calculate_adversary_loss(X_batch=X_batch, Z_batch=Z_batch))
            losses.update(self._calculate_encoder_loss(X_batch=X_batch, Z_batch=Z_batch))
            
            # print("calculated all the losses!!!!!!!!!!!!!!!!")
        return losses

    def _step(self, who=None):
        if who is not None:
            self.optimizers[who].step()
            if who == "Adversary":
                if self.adv_type == "Critic":
                    for p in self.adversary.parameters():
                        p.data.clamp_(-0.01, 0.01)
        else:
            [optimizer.step() for _, optimizer in self.optimizers.items()]


    #########################################################################
    # Utility functions
    #########################################################################
    def encode(self, x):
        return self.encoder(x)


    def generate_(self, X_batch=None, Z_batch=None):
        mu = None
        logvar = None
        total_preds = None
        discrim_features = None
        if X_batch is not None:
            mu, logvar = self.encoder(X_batch)

            # do not detach, as we want grad to flow from the decoder into the encoder
            mu = mu
            logvar = logvar

            # we need the true variances, not the log vars
            variances = torch.exp(logvar * 0.5)
            #sample from a Guassian
            if self.training:
                sample_from_normal = torch.autograd.Variable(torch.randn((len(X_batch), np.prod(self.z_dim)), device=self.device), requires_grad=True)
            else:
                sample_from_normal = torch.autograd.Variable(torch.randn((len(X_batch), np.prod(self.z_dim)), device=self.device), requires_grad=False)

            # shift and scale using the means and variances
            latent_z_sample = sample_from_normal * variances + mu

            # use latent sample to generate an output image
            generated_images = self.generator(latent_z_sample)

            # now get the discriminator feature info
            # start by concatonating the reconstructed, real and fake images batch-wise
            generated_images_z = self.generator(Z_batch)
            # detach, as we don't want the gradient to flow into the generator

            #     the discriminator and the generator are trained entirely separately
            #     not the case for VAE component because encoder and decoder are "attached" and SHOULD grad into eachother
            dis_x_prime = torch.cat((generated_images.detach(), X_batch, generated_images_z.detach()), 0)

            # pass the whole thing to the discriminator
            total_preds = self.adversary(dis_x_prime)

            # get the feature activations
            discrim_features = self.adversary.network._get_feature_layer_activations()

        elif Z_batch is not None:
            # use normal sample to generate an output image
            generated_images = self.generator(Z_batch)
        
        else: # generate an image from a random sample
            #sample from a Guassian
            if self.training:
                sample_from_normal = torch.autograd.Variable(torch.randn((len(X_batch), np.prod(self.z_dim)), device=self.device), requires_grad=True)
            else:
                sample_from_normal = torch.autograd.Variable(torch.randn((len(X_batch), np.prod(self.z_dim)), device=self.device), requires_grad=False)

            # use normal sample to generate an output image
            generated_images= self.generator(sample_from_normal)
        
        return mu, logvar, generated_images, total_preds, discrim_features

    def generate_batch(self, X_batch, requires_grad=False):
        mu, logvar = self.encoder(X_batch)
        
        # we need the true variances, not the log vars
        variances = torch.exp(logvar * 0.5)

        sample_from_normal = torch.autograd.Variable(torch.randn((len(X_batch), np.prod(self.z_dim)), device=self.device), requires_grad=requires_grad)
        
        # shift and scale using the means and variances
        latent_z_sample = sample_from_normal * variances + mu
        
        # use latent sample to generate an output image
        generated_images = self.generator(latent_z_sample)

        return generated_images


    
    def _save_models(self, epoch, name=None):
        pass

    def _load_models(self, path=None, who=None, training=False):
        pass