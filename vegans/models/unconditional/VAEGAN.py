"""
VAEGAN
------
Implements the Variational Autoencoder Generative Adversarial Network[1].

Trains on Kullback-Leibler loss for the latent space and attaches a adversary to get better quality output.
The Decoder acts as the generator.

Losses:
    - Encoder: Kullback-Leibler
    - Generator / Decoder: Binary cross-entropy
    - Adversary: Binary cross-entropy
Default optimizer:
    - torch.optim.Adam
Custom parameter:
    - lambda_KL: Weight for the encoder loss computing the Kullback-Leibler divergence in the latent space.
    - lambda_x: Weight for the reconstruction loss of the real x dimensions.

References
----------
.. [1] https://arxiv.org/pdf/1512.09300.pdf
"""

from random import random
import torch

import numpy as np
import torch.nn as nn
import os

from torch.nn import L1Loss
from vegans.utils.layers import LayerReshape
from vegans.models.unconditional.AbstractGANGAE import AbstractGANGAE

class VAEGAN(AbstractGANGAE):
    """
    Parameters
    ----------
    generator: nn.Module
        Generator architecture. Produces output in the real space.
    adversary: nn.Module
        Adversary architecture. Produces predictions for real and fake samples to differentiate them.
    encoder: nn.Module
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
    lambda_KL: float
        Weight for the encoder loss computing the Kullback-Leibler divergence in the latent space.
    lambda_x: float
        Weight for the reconstruction loss of the real x dimensions.
    adv_type: "Discriminator", "Critic" or "Autoencoder"
        Indicating which adversarial architecture will be used.
    feature_layer : torch.nn.*
        Output layer used to compute the feature loss. Should be from either the discriminator or critic.
        If `feature_layer` is not None, the original generator loss is replaced by a feature loss, introduced
        [here](https://arxiv.org/abs/1606.03498v1).
    fixed_noise_size : int
        Number of images shown when logging. The fixed noise is used to produce the images in the folder/images
        subdirectory, the tensorboard images tab and the samples in get_training_results().
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
            lambda_KL=10,
            lambda_x=10,
            adv_type="Discriminator",
            feature_layer=None,
            fixed_noise_size=32,
            device=None,
            ngpu=0,
            folder="./veganModels/VAEGAN",
            secure=True,
            lr_decay=0.9,
            T0=5,
            T_mult=1,
            eta_min=1e-6,
            warm_steps=5,
            cycle=5):


        super().__init__(
            generator=generator, adversary=adversary, encoder=encoder,
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, adv_type=adv_type, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, ngpu=ngpu, folder=folder, secure=secure, lr_decay=lr_decay, T0=T0,
            T_mult=T_mult, eta_min=eta_min
        )

        self.lambda_KL = lambda_KL
        self.lambda_x = lambda_x
        self.hyperparameters["lambda_KL"] = lambda_KL
        self.hyperparameters["lambda_x"] = lambda_x
        self.cycle = cycle
        self.last_reset_epoch = 0

        # HACK - done by hand... do so automatically later
        self.lambda_KL_max = 0.9                    # maximum value we will allow the kl weight to reach  
        self.steps_per_epoch = 2696                 # number of steps per epoch (size of dataset / batch_size)
        self.warmup_steps = warm_steps              # number of epochs we want the kl to "warm up" or increase


        if self.secure:
            # TODO
            # if self.encoder.output_size == self.z_dim:
            #     raise ValueError(
            #         "Encoder output size is equal to z_dim, but for VAE algorithms the encoder last layers for mu and sigma " +
            #         "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
            #     ) T0, T_mult
            assert (self.generator.output_size == self.x_dim), (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )


    def _define_loss(self):
        loss_functions = super()._define_loss()
        loss_functions.update({"Reconstruction": nn.MSELoss()})
        return loss_functions

    #########################################################################
    # Actions during training
    #########################################################################

    # Good!
    #    Updating with maximization trick -> maximize log(D(G(z))) instead of min log(1 - D(G(Z)))
    def _calculate_generator_loss(self, X_batch, Z_batch, fake_images_x=None, fake_images_z=None):
        
        # based on Pytorch - DCGAN example
        self._zero_grad(who="Generator")

        ## Calculate the predication (GAN) based loss

        # generate the images
        fake_x = self.generate_batch(X_batch=X_batch, requires_grad=True)

        # generate the predictions for these images
        pred_fake_x = self.adversary(fake_x)

        # for future use
        fake_x_features = self.adversary.network._get_feature_layer_activations().detach()

        gen_loss_fake_x = self.loss_functions["Generator"](
            pred_fake_x, torch.ones_like(pred_fake_x, requires_grad=True)
        )

        # calculate / accumulate the loss
        gen_loss_fake_x.backward(retain_graph=True)

        # generate fake_z images
        fake_z = self.generator(Z_batch)
        pred_fake_z = self.adversary(fake_z)

        # for future use
        fake_z_features = self.adversary.network._get_feature_layer_activations().detach()

        gen_loss_fake_z = self.loss_functions["Generator"](
            pred_fake_z, torch.ones_like(pred_fake_z, requires_grad=True)
        )

        # calculate / accumulate loss
        gen_loss_fake_z.backward(retain_graph=True)
        ## end GAN loss calculations

        ## calculate reconstruction loss
        # get the discriminator features
        _ = self.adversary(X_batch.detach())
        real_x_features = self.adversary.network._get_feature_layer_activations()

        # calculate the reconstruction loss for the fake_x features
        recon_loss_x = (self.lambda_x)*self.loss_functions["Reconstruction"](real_x_features, fake_x_features)

        # accumulate gradient
        recon_loss_x.backward(retain_graph=True)
        # calcualte the reconstruction loss for the fake_z features
        recon_loss_z = (self.lambda_x)*self.loss_functions["Reconstruction"](real_x_features, fake_z_features)

        # accumulate gradient
        recon_loss_z.backward(retain_graph=True)

        #    add to get the total loss
        loss_decoder = gen_loss_fake_x + gen_loss_fake_z

        total_recon_loss = recon_loss_x + recon_loss_z

        gen_loss = loss_decoder + total_recon_loss

        # print("Generator Loss:")
        # print(gen_loss)

        return {
            "Generator": gen_loss,
            "Generator_x": gen_loss_fake_x,
            "Generator_z": gen_loss_fake_z,
            "Reconstruction_Gen": total_recon_loss
        }

    def _calculate_encoder_loss(self, X_batch, Z_batch, fake_images_x=None):
        
        # check KL weight cycle stuff
        if self.epoch_ctr_ % self.cycle == 0 and self.epoch_ctr_ != 0 and self.last_reset_epoch != self.epoch_ctr_:
            # reset KL divergence to small value 
            self.lambda_KL = 0.01
            self.last_reset_epoch = self.epoch_ctr_
            print("----- Resetting KL weight!!!! %.4f -----" % self.lambda_KL)


        # based on Pytorch - DCGAN example
        self._zero_grad(who="Encoder")

        # use the encoder to get the values
        mu, logvar = self.encoder(X_batch)

        # get the actual prob instead of log-prob
        variances = torch.exp(logvar * 0.5)
        
        # do the reparam trick
        sample_from_normal = torch.randn((len(X_batch), np.prod(self.z_dim)), device=self.device, requires_grad=True)
        latent_z_sample = sample_from_normal * variances + mu

        # get the reconstructed images
        recon_images = self.generator(latent_z_sample)

        # calcualte the KL between the latent dist and the prior dist
        kl_loss = torch.sum(0.5*(logvar.exp() + mu**2 - logvar - 1), 1)

        kl_loss = self.lambda_KL*torch.mean(kl_loss, 0) 

        # calculate the preds for the reconstructed input
        pred_fake_x = self.adversary(recon_images)
        fake_x_features = self.adversary.network._get_feature_layer_activations().detach()

        # only used for reporting, not learning
        enc_loss_fake_x = self.loss_functions["Encoder"](
            pred_fake_x.detach(), torch.zeros_like(pred_fake_x.detach(), requires_grad=False).detach()
        )

        # get the preds for a randomly generate set of samples
        fake_z_images = self.generator(Z_batch)
        _ = self.adversary(fake_z_images)
        fake_z_features = self.adversary.network._get_feature_layer_activations()

        # get the features for the unmodified input
        _ = self.adversary(X_batch)
        real_x_features = self.adversary.network._get_feature_layer_activations()

        # calculate reconstruction loss values
        # calculate the reconstruction loss for the fake_x features
        recon_loss_x = self.loss_functions["Reconstruction"](real_x_features, fake_x_features)

        # accumulate gradient
        recon_loss_x.backward(retain_graph=True)

        # calcualte the reconstruction loss for the fake_z features
        recon_loss_z = self.loss_functions["Reconstruction"](real_x_features, fake_z_features)

        # accumulate gradient
        recon_loss_z.backward(retain_graph=True)

        total_recon_loss = recon_loss_x + recon_loss_z
        
        # The authors do not apply the KL lambda weight to the encoder
        enc_loss = (kl_loss + total_recon_loss)

        # print("Encoder Loss")
        # print(enc_loss)

        # update the lambda kl value
        self.lambda_KL = min(self.lambda_KL_max, self.lambda_KL + 1.0 / (self.warmup_steps * self.steps_per_epoch))

        # print(self.lambda_KL)

        return {
            "Encoder": enc_loss,
            "Encoder_x": enc_loss_fake_x,
            "Kullback-Leibler": kl_loss,
            "Reconstruction_Enc": total_recon_loss
        }

    def _calculate_adversary_loss(self, X_batch, Z_batch, fake_images_x=None, fake_images_z=None):

        # based on Pytorch - DCGAN example
        self._zero_grad(who="Adversary")

        # calculate stats using real batch data
        real_preds = self.adversary(X_batch)

        # calculate the loss on the real data
        adv_loss_real = self.loss_functions["Adversary"](
            real_preds, self.clean_discriminatoir_labels(torch.ones_like(real_preds, requires_grad=True))
        )

        # step this loss backwards
        adv_loss_real.backward(retain_graph=True)

        # generate images based on input and z's
        fake_z = self.generator(Z_batch.detach())
        fake_x = self.generate_batch(X_batch=X_batch, requires_grad=True)

        fake_z = fake_z.detach()
        fake_x = fake_x.detach()


        # now train with reconstructed-real data
        fake_preds_x = self.adversary(fake_x)

        # calculate the loss value
        adv_loss_fake_x = self.loss_functions["Adversary"](
            fake_preds_x, torch.zeros_like(fake_preds_x, requires_grad=False)
        )

        # calculate gradients for this batch, accumulate with previous
        adv_loss_fake_x.backward(retain_graph=True)

        # now train with all-fake data
        #    detaching based on DCGAN example-code
        fake_preds_z = self.adversary(fake_z)

        # calculate the loss value
        adv_loss_fake_z = self.loss_functions["Adversary"](
            fake_preds_z, torch.zeros_like(fake_preds_z, requires_grad=False)
        )
        
        # update the loss value for this batch, accumulate graident with previous
        adv_loss_fake_z.backward(retain_graph=True)

        adv_loss = (adv_loss_fake_z + adv_loss_fake_x + adv_loss_real)

        # print("Discriminator Loss:")
        # print(adv_loss)

        return {
            "Adversary": adv_loss,
            "Adversary_fake_x": adv_loss_fake_x,
            "Adversary_fake_z": adv_loss_fake_z,
            "Adversary_real": adv_loss_real,
            "RealFakeRatio": adv_loss_real / adv_loss_fake_x
        }

    # Perform label-smoothing - suggested GAN hack - https://github.com/soumith/ganhacks (source for idea) https://arxiv.org/pdf/1701.00160.pdf (paper proposing idea)
    def label_smoothing_(self, y):
        rand_vec = torch.rand(y.size())
        rand_vec = rand_vec.to(device=self.device)
        temp_ = y - 0.3 + (rand_vec*0.5)              # labels between [0.7, 1.2]
        temp_[temp_ < 0] = 0                          # can't have negative label values, unnecessary if doing one-sided label smoothing
        return temp_


    # randomly flip small protion of discriminator labels - suggested GAN hack - https://github.com/soumith/ganhacks, NIPS 2016 tutorial + Soumith
    def nosiy_labels_(self, y, p_flip):
        # determine the number of labels to flip
        n_select = int(p_flip * y.size()[0])
        # choose labels to flip
        flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
        # invert the labels in place
        y[flip_ix] = 1 - y[flip_ix]
        return y


    def clean_discriminatoir_labels(self, y):
        y_ = self.label_smoothing_(y)
        return self.nosiy_labels_(y_, p_flip=0.05)    # 5% chance of a flip


    def _save_models(self, epoch, name=None):
        """ Saves model in the model folder as torch / pickle object.

        Parameters
        ----------
        name : str, optional
            name of the saved file. folder specified in the constructor used
            in absolute path.
        """
        if name is None:
            name = "model.torch"
        if self.folder is not None:
            torch.save({'encoder' : self.neural_nets["Encoder"].network.state_dict(),
                        'decoder' : self.neural_nets["Generator"].network.state_dict(),
                        "discriminator" : self.neural_nets["Adversary"].network.state_dict(),
                        "encoder_opt_state" : self.optimizers["Encoder"].state_dict(),
                        "decoder_opt_state" : self.optimizers["Generator"].state_dict(),
                        "discriminator_opt_state" : self.optimizers["Adversary"].state_dict(),
                        # "encoder_lr_state" : self.lr_schedulers["Encoder"].state_dict(),
                        # "decoder_lr_state" : self.lr_schedulers["Generator"].state_dict(),
                        "discriminator_lr_state" : self.lr_schedulers["Adversary"].state_dict(),
                        "epoch" : epoch,
                        "losses" : self._losses
                        },
                        os.path.join(self.folder, name))

        else:
            torch.save({'encoder' : self.neural_nets["Encoder"].network.state_dict(),
                        'decoder' : self.neural_nets["Generator"].network.state_dict(),
                        "discriminator" : self.neural_nets["Adversary"].network.state_dict(),
                        "encoder_opt_state" : self.optimizers["Encoder"].state_dict(),
                        "decoder_opt_state" : self.optimizers["Generator"].state_dict(),
                        "discriminator_opt_state" : self.optimizers["Adversary"].state_dict(),
                        # "encoder_lr_state" : self.lr_schedulers["Encoder"].state_dict(),
                        # "decoder_lr_state" : self.lr_schedulers["Generator"].state_dict(),
                        "discriminator_lr_state" : self.lr_schedulers["Adversary"].state_dict(),
                        "epoch" : epoch,
                        "losses" : self._losses
                        },
                        os.path.join("", name))

   
    def _load_models(self, name=None, training=False):
        # assign a default model file name
        if name is None:
            name = "model.torch"

        # construct the path to the saved model
        path = ""
        if self.folder is not None:
            path = os.path.join(self.folder, name)
        else:
            path = os.path.join(self.folder, name)
        
        # load the saved checkpoint
        check_point = torch.load(path)

        # load the epoch we saved this model from
        epoch_ = check_point["epoch"]

        # load the actual models
        self.neural_nets["Encoder"].network.load_state_dict(check_point["encoder"])
        self.neural_nets["Generator"].network.load_state_dict(check_point["decoder"])
        self.neural_nets["Adversary"].network.load_state_dict(check_point["discriminator"])
        
        # if we are loading the model to continue training
        if training:
            # load the optimizers
            self.optimizers["Encoder"].load_state_dict(check_point["encoder_opt_state"])
            self.optimizers["Generator"].load_state_dict(check_point["decoder_opt_state"])
            self.optimizers["Adversary"].load_state_dict(check_point["discriminator_opt_state"])

            # load the learning rate schedulers
            self.lr_schedulers["Encoder"].load_state_dict(check_point["encoder_lr_state"])
            self.lr_schedulers["Generator"].load_state_dict(check_point["decoder_lr_state"])
            self.lr_schedulers["Adversary"].load_state_dict(check_point["discriminator_lr_state"])

            # load the loss dict
            self._losses = check_point["losses"]

        # return the epoch this model was saved at
        return epoch_
