from contextlib import contextmanager
import cv2
import torch
import numpy as np
import torch.nn as nn

from tarp.components.base_model import BaseModel
from tarp.modules.losses import L2Loss, KLDivLoss
from tarp.modules.subnetworks import Encoder, Decoder, Predictor
from tarp.utils.general_utils import AttrDict, ParamDict, remove_spatial, batch_apply, get_clipped_optimizer
from tarp.utils.pytorch_utils import TensorModule, RAdam, ar2ten, ten2ar
from tarp.modules.variational_inference import ProbabilisticModel, MultivariateGaussian, get_fixed_prior
from tarp.modules.layers import LayerBuilderParams
from tarp.modules.recurrent_modules import BaseProcessingLSTM, RecurrentPredictor
from tarp.utils.vis_utils import make_gif_strip, make_image_strip, make_image_seq_strip


class PredictionMdl(BaseModel, ProbabilisticModel):
    """Simple recurrent forward predictor network with image encoder and decoder."""
    def __init__(self, params, logger):
        BaseModel.__init__(self, logger)
        ProbabilisticModel.__init__(self)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)

        # set up beta tuning (use fixed beta by default)
        if self._hp.target_kl is None:
            self._log_beta = TensorModule(np.log(self._hp.fixed_beta)
                                          * torch.ones(1, requires_grad=False, device=self._hp.device))
        else:
            self._log_beta = TensorModule(torch.zeros(1, requires_grad=True, device=self._hp.device))
            self.beta_opt = self._get_beta_opt()

        self._task_names = [task_name().name if not isinstance(task_name, str) else task_name
                            for task_name in self._hp.task_names]

        self.build_network()

    @contextmanager
    def val_mode(self):
        self.switch_to_prior()
        yield
        self.switch_to_inference()
    
    def _default_hparams(self):
        # put new parameters in here:
        default_dict = ParamDict({
            'use_skips': True,
            'skips_stride': 1,
            'add_weighted_pixel_copy': False,  # if True, adds pixel copying stream for decoder
            'pixel_shift_decoder': False,
            'use_convs': True,
            'normalization': 'batch',
            'predict_rewards': False,       # if True, predicts future rewards in addition to observations
        })

        # Network size
        default_dict.update({
            'img_sz': 32,  # image resolution
            'input_nc': 3,  # number of input feature maps
            'ngf': 4,  # number of feature maps in shallowest level
            'nz_enc': 32,  # number of dimensions in encoder-latent space
            'nz_vae': 32,  # number of dimensions in vae-latent space
            'nz_mid': 32,  # number of dimensions for internal feature spaces
            'nz_mid_lstm': 32,  # size of middle LSTM layers
            'n_lstm_layers': 1,  # number of LSTM layers
            'n_processing_layers': 3,  # Number of layers in MLPs
        })

        # Loss weights
        default_dict.update({
            'img_mse_weight': 1.,
            'fixed_beta': 1.0,
            'target_kl': None,
            'reward_pred_mse_weight': 1.0,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params
    
    def build_network(self):
        self.q = self._build_inference_net()
        self.q_enc = Encoder(self._hp)
        self.encoder = Encoder(self._hp)
        self.enc2z = torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae*2)
        self.predictor = RecurrentPredictor(self._hp,
                                            input_size=self._hp.nz_vae+self._hp.nz_enc,
                                            output_size=self._hp.nz_enc)
        self.decoder = Decoder(self._hp)
        self.predictor_hidden_initalizer = self._build_predictor_initializer(size=self.predictor.cell.get_state_size())

        if self._hp.predict_rewards:
            self.reward_decoders = nn.ModuleDict({name: Predictor(self._hp, input_size=self._hp.nz_enc, output_size=1,
                                                                  spatial=False) for name in self._task_names})
    
    def forward(self, inputs):
        """
        forward pass at training time
        """
        output = AttrDict()

        # run inference
        enc_seq, _ = batch_apply(inputs.images, self.q_enc)
        output.q = MultivariateGaussian(self.q(remove_spatial(enc_seq))[:, -1])

        # encode conditioning input
        enc_input, skips = self.encoder(inputs.images[:, 0])

        # sample latent variable
        if self._sample_prior:
            output.z = get_fixed_prior(output.q).sample()
        else:
            output.z = output.q.rsample()

        # predict future embeddings
        pred_seq = self.predictor(lstm_initial_inputs=AttrDict(x_t=enc_input),
                                  lstm_static_inputs=AttrDict(z=output.z),
                                  steps=inputs.images.shape[1] - 1,
                                  lstm_hidden_init=self.predictor_hidden_initalizer(inputs.images)).pred

        # decode predictions
        output.output_imgs = self.decoder.decode_seq(AttrDict(skips=skips), pred_seq).images

        # (optionally) decode rewards
        if self._hp.predict_rewards:
            output.r_hat = AttrDict({name: batch_apply(pred_seq, self.reward_decoders[name])[..., 0]
                                     for name in self._task_names})

        # store some fields for evaluation
        inputs.observations = inputs.images[:, 1:]
        output.reconstruction = output.output_imgs
        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # reconstruction loss
        losses.img_mse = L2Loss(self._hp.img_mse_weight)(model_output.output_imgs, inputs.images[:, 1:])

        # KL loss
        losses.kl_loss = KLDivLoss(self.beta[0].detach())(model_output.q, get_fixed_prior(model_output.q))

        # (optionally) add reward prediction loss
        if self._hp.predict_rewards:
            losses.update(AttrDict({name: L2Loss(self._hp.reward_pred_mse_weight)(model_output.r_hat[name][inputs.task_id==i],
                                                                                  inputs.rewards[:, 1:][inputs.task_id==i])
                                    for i, name in enumerate(self._task_names)}))

        # Update Beta
        if self.training:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)
        self._logger.log_scalar(self.beta[0], 'beta', step, phase)
        # if log_images:
        #     # log reconstructions / prior samples
        #     gif_strip = make_gif_strip([inputs.images[:, 1:, :1], model_output.output_imgs[:, :, :1]])
        #     gif_strip = torch.cat(3*[gif_strip], dim=1)     # make RGB
        #     self._logger.log_gif(gif_strip[None], "prediction", step, phase, fps=5)

            # attention mask
            # self._log_attention_mask(inputs, step, phase)

    def _build_inference_net(self):
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=self._hp.nz_enc, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )

    def _build_predictor_initializer(self, size):
        class FixedTrainableInitializer(nn.Module):
            def __init__(self, hp):
                super().__init__()
                self._hp = hp
                self.val = nn.Parameter(torch.zeros((1, size), requires_grad=True, device=self._hp.device))

            def forward(self, state):
                return self.val.repeat(state.shape[0], 1)
        return FixedTrainableInitializer(self._hp)

    def _get_beta_opt(self):
        return get_clipped_optimizer(filter(lambda p: p.requires_grad, self._log_beta.parameters()),
                                     lr=3e-4, optimizer_type=RAdam, betas=(0.9, 0.999), gradient_clip=None)

    def _update_beta(self, kl_div):
        """Updates beta with dual gradient descent."""
        if self._hp.target_kl is not None:
            beta_loss = self.beta * (self._hp.target_kl - kl_div).detach().mean()
            self.beta_opt.zero_grad()
            beta_loss.backward()
            self.beta_opt.step()

    def forward_encoder(self, inputs):
        enc = self.encoder(inputs)
        return enc

    @property
    def resolution(self):
        return self._hp.img_sz

    @property
    def beta(self):
        return self._log_beta().exp()
