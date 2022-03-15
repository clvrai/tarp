from contextlib import contextmanager
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
from copy import deepcopy

from tarp.components.base_model import BaseModel
from tarp.modules.losses import L2Loss, BCELoss, CELoss, BCEWithLogitsLoss
from tarp.modules.subnetworks import Encoder, Decoder, Predictor
from tarp.modules.recurrent_modules import RecurrentPredictor
from tarp.utils.general_utils import AttrDict, ParamDict, batch_apply
from tarp.utils.pytorch_utils import pad_seq, make_one_hot, ar2ten, ten2ar
from tarp.modules.layers import LayerBuilderParams
from tarp.utils.vis_utils import make_gif_strip, make_image_seq_strip, make_image_strip

class TARPHeterogeneousModel(BaseModel):
    def __init__(self, params, logger):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)
        self._task_names = [task_name().name if not isinstance(task_name, str) else task_name
                              for task_name in self._hp.task_names]

        self.build_network()

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass

    def _default_hparams(self):
        default_dict = ParamDict({
            'use_skips': False,
            'skips_stride': 2,
            'add_weighted_pixel_copy': False, # if True, adds pixel copying stream for decoder
            'pixel_shift_decoder': False,
            'use_convs': True,
            'detach_reconstruction': True,
            'detach_predictor': False,
            'n_cond_frames': 1,
            'n_class': 1,
            'n_action': 1,
            'normalization': 'none',
            'action_space_type': 'discrete',
            'use_seg_mask': False,
        })

        # Network size
        default_dict.update({
            'img_sz': 32,
            'input_nc': 3,
            'ngf': 8,
            'nz_enc': 32,
            'nz_mid': 32,
            'n_processing_layers': 3,
            'n_pixel_sources': 1,
        })

        # Loss weights
        default_dict.update({
            'img_mse_weight': 1.,
            'value_weights': 1.,
            'action_weights': 1.
        })

        # model specific parameter
        default_dict.update({
            'data_source_maps': AttrDict() # mapping between task and data type (e.g. value, action)
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        self.encoder = Encoder(self._hp)
        self.decoder = Decoder(self._hp)

        self.predictors = nn.ModuleDict()
        modules = {}
        for key, data_type in self._hp.data_source_maps.items():
            if data_type == 'action':
                modules[key] = Predictor(self._hp, input_size=self._hp.nz_enc,
                                         output_size=self._hp.n_action, spatial=False)
            elif data_type == 'value':
                modules[key] = Predictor(self._hp, input_size=self._hp.nz_enc,
                                         output_size=1, spatial=False)
        self.predictors.update(modules)

    def forward(self, inputs):
        """
        forward pass at training time
        """
        output = AttrDict()

        # encode inputs
        enc = self.encoder(inputs.images[:, 0])
        output.update({'pred': enc, 'rec_input': enc})


        rec_input = output.rec_input.detach() if self._hp.detach_reconstruction else output.rec_input
        output.output_imgs = self.decoder(rec_input).images.unsqueeze(1)


        predictor_input = output.pred.detach() if self._hp.detach_predictor else output.pred
        output.output_values = AttrDict({name: self.predictors[name](predictor_input) for name in self._task_names})

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # image reconstruction loss
        losses.seq_img_mse = L2Loss(self._hp.img_mse_weight)(model_output.output_imgs,
                                                             inputs.images[:, :1])

        for i, name in enumerate(self._task_names):
            data_type = self._hp.data_source_maps[name]
            if data_type == 'action':
                if self._hp.action_space_type == 'discrete':
                    target_actions = make_one_hot(inputs.actions[:, 0, 0].long(), self._hp.n_action).type(torch.float)
                    losses.update(AttrDict({'action_'+name: BCEWithLogitsLoss(self._hp.action_weights)(model_output.output_values[name][inputs.task_id==i],
                                                                                target_actions[inputs.task_id==i])}))
                elif self._hp.action_space_type == 'continuous':
                    losses.update(AttrDict({'action_'+name: L2Loss(self._hp.action_weights)(model_output.output_values[name][inputs.task_id==i],
                                                                     inputs.actions[:, 0][inputs.task_id==i])}))
                else:
                    raise NotImplementedError
            elif data_type == 'value':
                losses.update(AttrDict({'value_'+name: L2Loss(self._hp.value_weights)(model_output.output_values[name][inputs.task_id==i], inputs.discounted_returns[:, :1][inputs.task_id==i])}))


        losses.total = self._compute_total_loss(losses)
        return losses

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)
        if log_images:
            # log predicted images
            img_strip = make_image_strip([inputs.images[:, 0, -int(self._hp.input_nc//self._hp.n_frames):],
                                          model_output.output_imgs[:, 0, -int(self._hp.input_nc//self._hp.n_frames):]])
            self._logger.log_images(img_strip[None], 'generation', step, phase)

        # attention mask
        self._log_attention_mask(inputs, step, phase)

    def forward_encoder(self, inputs):
        enc = self.encoder(inputs)
        return enc

    @property
    def resolution(self):
        return self._hp.img_sz


class TARPRecurrentHeterogeneousModel(BaseModel):
    def _default_hparams(self):
        default_dict = ParamDict({
            'nz_mid_lstm': 512,
        })
        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        self.encoder = Encoder(self._hp)
        self.decoder = Decoder(self._hp)
        self.lstm = nn.LSTM(self._hp.nz_enc, self._hp.nz_mid_lstm, bidirectional=False, batch_first=True)

        self.predictors = nn.ModuleDict()
        modules = {}
        for key, data_type in self._hp.data_source_maps.items():
            if data_type == 'action':
                modules[key] = Predictor(self._hp, input_size=self._hp.nz_enc,
                                         output_size=self._hp.n_action, spatial=False)
            elif data_type == 'value':
                modules[key] = Predictor(self._hp, input_size=self._hp.nz_enc,
                                         output_size=1, spatial=False)
        self.predictors.update(modules)

    def forward(self, inputs):
        """
        forward pass at training time
        """
        output = AttrDict()

        enc = batch_apply(inputs.images, self.encoder)
        output.update({'pred': enc.view(enc.shape[0]*enc.shape[1], *enc.shape[2:]), 'rec_input': enc.view(enc.shape[0]*enc.shape[1], *enc.shape[2:])})
        h_t, _ = self.lstm(enc.squeeze())

        rec_input = output.rec_input.detach() if self._hp.detach_reconstruction else output.rec_input
        output.output_imgs = self.decoder(rec_input).images.unsqueeze(1)


        predictor_input = output.pred.detach() if self._hp.detach_predictor else output.pred
        output.output_values = AttrDict({name: self.predictors[name](predictor_input) for name in self._task_names})

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # image reconstruction loss
        losses.seq_img_mse = L2Loss(self._hp.img_mse_weight)(model_output.output_imgs,
                inputs.images.view(inputs.images.shape[0]*inputs.images.shape[1],1,*inputs.images.shape[2:]))

        inputs.task_id = inputs.task_id.unsqueeze(1).repeat(1, inputs.action.shape[1]).reshape(-1)
        for i, name in enumerate(self._task_names):
            data_type = self._hp.data_source_maps[name]
            if data_type == 'action':
                if self._hp.action_space_type == 'discrete':
                    target_actions = make_one_hot(inputs.actions[:, 0, 0].long(), self._hp.n_action).type(torch.float)
                    losses.update(AttrDict({'action_'+name: BCEWithLogitsLoss(self._hp.action_weights)(model_output.output_values[name][inputs.task_id==i],
                                                                                target_actions[inputs.task_id==i])}))
                elif self._hp.action_space_type == 'continuous':
                    losses.update(AttrDict({'action_'+name: L2Loss(self._hp.action_weights)(model_output.output_values[name][inputs.task_id==i],
                                                                 inputs.action.view(inputs.action.shape[0]*inputs.action.shape[1],*inputs.action.shape[2:])[inputs.task_id==i])}))
                else:
                    raise NotImplementedError
            elif data_type == 'value':
                losses.update(AttrDict({'value_'+name: L2Loss(self._hp.value_weights)(model_output.output_values[name][inputs.task_id==i],
                                                                                      inputs.discounted_returns.view(inputs.discounted_returns.shape[0]*inputs.discounted_returns.shape[1], 1)[inputs.task_id==i])}))


        losses.total = self._compute_total_loss(losses)
        return losses
