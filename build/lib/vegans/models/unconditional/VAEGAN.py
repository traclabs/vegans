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
            secure=True):


        super().__init__(
            generator=generator, adversary=adversary, encoder=encoder,
            x_dim=x_dim, z_dim=z_dim, optim=optim, optim_kwargs=optim_kwargs, adv_type=adv_type, feature_layer=feature_layer,
            fixed_noise_size=fixed_noise_size, device=device, ngpu=ngpu, folder=folder, secure=secure
        )

        self.lambda_KL = lambda_KL
        self.lambda_x = lambda_x
        self.hyperparameters["lambda_KL"] = lambda_KL
        self.hyperparameters["lambda_x"] = lambda_x

        if self.secure:
            # TODO
            # if self.encoder.output_size == self.z_dim:
            #     raise ValueError(
            #         "Encoder output size is equal to z_dim, but for VAE algorithms the encoder last layers for mu and sigma " +
            #         "are constructed by the algorithm itself.\nSpecify up to the second last layer for this particular encoder."
            #     )
            assert (self.generator.output_size == self.x_dim), (
                "Generator output shape must be equal to x_dim. {} vs. {}.".format(self.generator.output_size, self.x_dim)
            )


    def _define_loss(self):
        loss_functions = super()._define_loss()
        loss_functions.update({"Reconstruction": nn.MSELoss()})
        return loss_functions


    def calculate_common_loss_values_(self, X_batch, Z_batch):
        bs_ = X_batch.size()[0]

        # generate images using the network(s)
        self.mu, self.logvar, self.fake_images_x, total_preds, discrim_feats = self.generate_(X_batch=X_batch, Z_batch=Z_batch)

        # get the stuff for the reconstructed images
        self.pred_fake_x = total_preds[:bs_]
        self.feat_fake_x = discrim_feats[:bs_]

        # get the stuff for the real images
        self.pred_real_x = total_preds[bs_:-bs_]
        self.feat_real_x = discrim_feats[bs_:-bs_]

        # get the stuff for sampled images
        self.pred_fake_z = total_preds[-bs_:]
        self.feat_fake_z = discrim_feats[-bs_:]

        # get_reconstruction_loss(self, X_batch, fake_images_x, fake_images_z):
        self.recon_loss_x, self.recon_loss_z = self.get_reconstruction_loss(self.feat_real_x, self.feat_fake_x, self.feat_fake_z)

        self.total_recon_loss = self.recon_loss_x + self.recon_loss_z

    #########################################################################
    # Actions during training
    #########################################################################
    # Good!
    # def _calculate_generator_loss(self, X_batch, Z_batch, fake_images_x=None, fake_images_z=None):
    #     # # # generate images using the network(s)
    #     # if fake_images_x is None:
    #     #     _, _, fake_images_x = self.generate_(X_batch=X_batch, Z_batch=Z_batch)

    #     # if fake_images_z is None:
    #     #     _, _, fake_images_z = self.generate_(Z_batch=Z_batch)

    #     # # calculate the reconstruction loss2
    #     # gen_loss_fake_x, gen_loss_fake_z = self.get_reconstruction_loss(X_batch, fake_images_x, fake_images_z)
    #     # gen_loss_reconstruction = torch.sum(gen_loss_fake_x) + torch.sum(gen_loss_fake_z)

    #     # # calculate the relavent parts of the GAN loss
    #     # #    get generated images
    #     # fake_predictions_z = self.predict(x=fake_images_z)
    #     # fake_predictions_x = self.predict(x=fake_images_x)
        
    #     #    calculate the loss function
    #     adv_loss_fake_x = self.loss_functions["Generator"](
    #         self.pred_fake_x, torch.zeros_like(self.pred_fake_x, requires_grad=False)
    #     )
        
    #     adv_loss_fake_z = self.loss_functions["Generator"](
    #         self.pred_fake_z, torch.zeros_like(self.pred_fake_z, requires_grad=False)
    #     )

    #     #    add to get the total loss
    #     loss_decoder = adv_loss_fake_x + adv_loss_fake_z

    #     gen_loss = loss_decoder + self.lambda_x*self.total_recon_loss

    #     print("Generator Loss:")
    #     print(gen_loss)

    #     return {
    #         "Generator": gen_loss,
    #         "Generator_x": self.recon_loss_x,
    #         "Generator_z": self.recon_loss_z,
    #         "Reconstruction": self.lambda_x*self.total_recon_loss
    #     }

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
        recon_loss_x = (self.lambda_x/2.0)*self.loss_functions["Reconstruction"](real_x_features, fake_x_features)

        # accumulate gradient
        recon_loss_x.backward(retain_graph=True)
        # calcualte the reconstruction loss for the fake_z features
        recon_loss_z = (self.lambda_x/2.0)*self.loss_functions["Reconstruction"](real_x_features, fake_z_features)

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
            "Reconstruction": total_recon_loss
        }

    # Good!
    # def _calculate_encoder_loss(self, X_batch, Z_batch, fake_images_x=None):
    #     # # encode the input and get fake images 
    #     # if fake_images_x is None:
    #     #     mu, log_variance, fake_images_x = self.generate_(X_batch=X_batch, Z_batch=Z_batch)

    #     # # get the fake z images
    #     # _, _, fake_images_z = self.generate_(Z_batch=Z_batch)

    #     # # get the reconstruction loss
    #     # gen_loss_fake_x, gen_loss_fake_z = self.get_reconstruction_loss(X_batch, fake_images_x, fake_images_z)
    #     # enc_loss_reconstruction = torch.sum(gen_loss_fake_x) + torch.sum(gen_loss_fake_z) # sum here might be redundant

    #     # fake_predictions_x = self.predict(x=fake_images_x)

    #     # only used for reporting, not learning
    #     enc_loss_fake_x = self.loss_functions["Encoder"](
    #         self.pred_fake_x, self.label_smoothing_(torch.ones_like(self.pred_fake_x, requires_grad=False))
    #     )

    #     # calcualte the KL between the latent dist and the prior dist
    #     kl_loss = torch.sum(0.5*(self.logvar.exp() + self.mu**2 - self.logvar - 1), 1) # good!
        
    #     # The authors do not apply the KL lambda weight to the encoder
    #     enc_loss = (self.lambda_KL*torch.sum(kl_loss) + self.lambda_x*self.total_recon_loss)

    #     print("Encoder Loss")
    #     print(enc_loss)

    #     return {
    #         "Encoder": enc_loss,
    #         "Encoder_x": enc_loss_fake_x,
    #         "Kullback-Leibler": self.lambda_KL*kl_loss,
    #         "Reconstruction": self.lambda_x*self.total_recon_loss
    #     }

    def _calculate_encoder_loss(self, X_batch, Z_batch, fake_images_x=None):
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

        kl_loss = self.lambda_KL*torch.sum(kl_loss) 

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
        recon_loss_x = (self.lambda_x/2.0)*self.loss_functions["Reconstruction"](real_x_features, fake_x_features)

        # accumulate gradient
        recon_loss_x.backward(retain_graph=True)

        # calcualte the reconstruction loss for the fake_z features
        recon_loss_z = (self.lambda_x/2.0)*self.loss_functions["Reconstruction"](real_x_features, fake_z_features)

        # accumulate gradient
        recon_loss_z.backward(retain_graph=True)

        total_recon_loss = recon_loss_x + recon_loss_z
        
        # The authors do not apply the KL lambda weight to the encoder
        enc_loss = (kl_loss + total_recon_loss)

        # print("Encoder Loss")
        # print(enc_loss)

        return {
            "Encoder": enc_loss,
            "Encoder_x": enc_loss_fake_x,
            "Kullback-Leibler": kl_loss,
            "Reconstruction": total_recon_loss
        }

    # Good!
    # def _calculate_adversary_loss(self, X_batch, Z_batch, fake_images_x=None, fake_images_z=None):
        
    #     # if fake_images_x is None:
    #     #     _, _, fake_images_x = self.generate_(X_batch=X_batch, Z_batch=Z_batch)

    #     # if fake_images_z is None:
    #     #     fake_images_z = self.generate_(Z_batch=Z_batch)[-1].detach()

    #     # fake_predictions_x = self.predict(x=fake_images_x)
    #     # fake_predictions_z = self.predict(x=fake_images_z)
    #     # real_predictions = self.predict(x=X_batch)

    #     adv_loss_fake_x = self.loss_functions["Adversary"](
    #         self.pred_fake_x, torch.zeros_like(self.pred_fake_x, requires_grad=False)
    #     )
        
    #     adv_loss_fake_z = self.loss_functions["Adversary"](
    #         self.pred_fake_z, torch.zeros_like(self.pred_fake_z, requires_grad=False)
    #     )
        
    #     adv_loss_real = self.loss_functions["Adversary"](
    #         self.pred_real_x, self.clean_discriminatoir_labels(torch.ones_like(self.pred_real_x, requires_grad=False))
    #     )

    #     adv_loss = (adv_loss_fake_z + adv_loss_fake_x + adv_loss_real)

    #     print("Discriminator Loss:")
    #     print(adv_loss)

    #     return {
    #         "Adversary": adv_loss,
    #         "Adversary_fake_x": adv_loss_fake_x,
    #         "Adversary_fake_z": adv_loss_fake_z,
    #         "Adversary_real": adv_loss_real,
    #         "RealFakeRatio": adv_loss_real / adv_loss_fake_x
    #     }

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


    def get_reconstruction_loss(self, X_batch, fake_images_x, fake_images_z):
        # calculate the reconstruction loss at the pixel-level (fall-back option)
        recon_loss_x = self.loss_functions["Reconstruction"](fake_images_x, X_batch)
        recon_loss_z = self.loss_functions["Reconstruction"](fake_images_z, X_batch)
        return recon_loss_x, recon_loss_z

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

   
    # TODO: Need to add generate from a sample input
