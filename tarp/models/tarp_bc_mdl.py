from contextlib import contextmanager
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
from copy import deepcopy

from tarp.components.base_model import BaseModel
from tarp.modules.losses import L2Loss, BCELoss, CELoss, BCEWithLogitsLoss, NLL
from tarp.modules.subnetworks import Encoder, Decoder, Predictor
from tarp.modules.recurrent_modules import RecurrentPredictor
from tarp.utils.general_utils import AttrDict, ParamDict, batch_apply
from tarp.utils.pytorch_utils import pad_seq, make_one_hot, ar2ten, ten2ar
from tarp.modules.layers import LayerBuilderParams
from tarp.utils.vis_utils import make_gif_strip, make_image_seq_strip, make_image_strip
from tarp.modules.variational_inference import MultivariateGaussian

class TARPBCModel(BaseModel):
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
            'detach_action_head': False,
            'n_cond_frames': 1,
            'detach_seg_mask': True,
            'seg_dec_activation': None,
            'n_class': 1,
            'n_action': 1,
            'normalization': 'none',
            'action_space_type': 'discrete',
            'use_seg_mask': False,
            'mask_sz': 300,
            'stochastic': False,
        })

        # Network size
        default_dict.update({
            'img_sz': 64,
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
            'reward_weights': 1.,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        parent_params.overwrite(default_dict)
        return parent_params

    def build_network(self):
        self.encoder = Encoder(self._hp)
        self.decoder = Decoder(self._hp)

        if self._hp.use_seg_mask:
            seg_decoder_hp = deepcopy(self._hp)
            seg_decoder_hp.input_nc = self._hp.n_class
            seg_decoder_hp.dec_last_activation = self._hp.seg_dec_activation
            self.seg_decoder = Decoder(seg_decoder_hp)

        self.action_heads = nn.ModuleDict({name: Predictor(self._hp, input_size=self._hp.nz_enc,
                                                          output_size=self.action_output_size, spatial=False) for name in self._task_names})

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

        if self._hp.use_seg_mask:
            seg_input = output.pred.detach() if self._hp.detach_seg_mask else output.pred
            output.output_seg = self.seg_decoder(seg_input).images.unsqueeze(1)

        action_input = output.pred.detach() if self._hp.detach_action_head else output.pred
        output.actions = AttrDict({name: self.action_heads[name](action_input) for name in self._task_names})

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # image reconstruction loss
        losses.seq_img_mse = L2Loss(self._hp.img_mse_weight)(model_output.output_imgs,
                                                             inputs.images[:, :1])

        if self._hp.use_seg_mask:
            losses.seq_seg_entropy = CELoss()(model_output.output_seg.reshape(self._hp.batch_size, self._hp.n_class, -1),
                                              inputs.seg_targets[:, :1].reshape(self._hp.batch_size, -1))

        if self._hp.action_space_type == 'discrete':
            target_actions = make_one_hot(inputs.actions[:, 0, 0].long(), self._hp.n_action).type(torch.float)
            losses.update(AttrDict({'action_'+name: BCEWithLogitsLoss()(model_output.actions[name][inputs.task_id==i],
                                                                        target_actions[inputs.task_id==i])
                                    for i, name in enumerate(self._task_names)}))
        elif self._hp.action_space_type == 'continuous':
            if self._hp.stochastic:
                losses.update(AttrDict({'action_' + name: NLL()(MultivariateGaussian(model_output.actions[name][inputs.task_id == i]),
                                                                   inputs.actions[:, 0][inputs.task_id == i])
                                        for i, name in enumerate(self._task_names)}))
            else:
                losses.update(AttrDict({'action_'+name: L2Loss()(model_output.actions[name][inputs.task_id==i],
                                                                 inputs.actions[:, 0][inputs.task_id==i])
                                        for i, name in enumerate(self._task_names)}))
        else:
            raise NotImplementedError


        # print(losses)
        losses.total = self._compute_total_loss(losses)
        return losses

    def weighted_mask_loss(self, pred_masks, gt_masks):
        bce = torch.nn.BCEWithLogitsLoss(reduction='none')(pred_masks, gt_masks)
        flipped_mask = self.flip_tensor(gt_masks)
        inside = (bce * gt_masks).sum() / ((gt_masks).sum()+1e-5)
        outside = (bce * flipped_mask).sum() / ((flipped_mask).sum()+1e-5)
        return inside + outside

    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        super()._log_losses(losses, step, log_images, phase)
        if log_images:
            # log predicted images
            img_strip = make_image_strip([inputs.images[:, 0, -int(self._hp.input_nc//self._hp.n_frames):],
                                          model_output.output_imgs[:, 0, -int(self._hp.input_nc//self._hp.n_frames):]])
            self._logger.log_images(img_strip[None], 'generation', step, phase)

            if self._hp.use_seg_mask:
                output_labels = torch.argmax(model_output.output_seg.squeeze(1), dim=1)
                b, _, nc, h, w = model_output.output_seg.shape
                pred_seg = torch.zeros((b, h, w, 3), device=self._hp.device)
                gt_seg = torch.zeros((b, h, w, 3), device=self._hp.device)
                for c in range(nc):
                    pred_seg[output_labels==c] = inputs.color_map[0].squeeze(0)[c].type(torch.float32)
                    gt_seg[inputs.seg_targets[:, :1].squeeze(1).squeeze(1)==c] = inputs.color_map[0].squeeze(0)[c].type(torch.float32)
                input_images = inputs.images[:, 0, -int(self._hp.input_nc//self._hp.n_frames):]
                if input_images.shape[1] == 1:
                    input_images = input_images.repeat((1, 3, 1, 1))
                mask_strip = make_image_strip([(input_images+1)*255/2,
                                                gt_seg.permute((0, 3, 1, 2)), pred_seg.permute((0, 3, 1, 2))])
                self._logger.log_images(mask_strip[None], 'segmentation', step, phase)
        # attention mask
        # self._log_attention_mask(inputs, step, phase)

    def forward_encoder(self, inputs):
        enc = self.encoder(inputs)
        return enc

    def special_state_dict(self):
        special_dict = super().special_state_dict()

        # also save first policy head
        special_dict.policy = self.action_heads[self._task_names[0]].state_dict()
        return special_dict

    @property
    def resolution(self):
        return self._hp.img_sz

    @property
    def action_output_size(self):
        if self._hp.stochastic and self._hp.action_space_type == 'continuous':
            return self._hp.n_action * 2
        else:
            return self._hp.n_action


class TARPRecurrentBCModel(TARPBCModel):
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

        if self._hp.use_seg_mask:
            seg_decoder_hp = deepcopy(self._hp)
            seg_decoder_hp.input_nc = self._hp.n_class
            seg_decoder_hp.dec_last_activation = self._hp.seg_dec_activation
            self.seg_decoder = Decoder(seg_decoder_hp)

        self.action_heads = nn.ModuleDict({name: Predictor(self._hp, input_size=self._hp.nz_mid_lstm,
                                                          output_size=self._hp.n_action, spatial=False) for name in self._task_names})

    def forward(self, inputs):
        """
        forward pass at training time
        """
        output = AttrDict()

        # encode inputs
        enc = batch_apply(inputs.images, self.encoder)
        output.update({'pred': enc.view(enc.shape[0]*enc.shape[1], *enc.shape[2:]), 'rec_input': enc.view(enc.shape[0]*enc.shape[1], *enc.shape[2:])})
        h_t, _ = self.lstm(enc.squeeze())
        action_input = h_t.reshape(h_t.shape[0]*h_t.shape[1], *h_t.shape[2:]).unsqueeze(2).unsqueeze(2)
        output.actions = AttrDict({name: self.action_heads[name](action_input) for name in self._task_names})


        rec_input = output.rec_input.detach() if self._hp.detach_reconstruction else output.rec_input
        output.output_imgs = self.decoder(rec_input).images.unsqueeze(1)

        if self._hp.use_seg_mask:
            seg_input = output.pred.detach() if self._hp.detach_seg_mask else output.pred
            output.output_seg = self.seg_decoder(seg_input).images.unsqueeze(1)

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()
        # image reconstruction loss
        losses.seq_img_mse = L2Loss(self._hp.img_mse_weight)(model_output.output_imgs,
                inputs.images.view(inputs.images.shape[0]*inputs.images.shape[1],1,*inputs.images.shape[2:]))

        if self._hp.use_seg_mask:
            losses.seq_seg_entropy = CELoss()(model_output.output_seg.reshape(self._hp.batch_size, self._hp.n_class, -1),
                                              inputs.seg_targets[:, :1].reshape(self._hp.batch_size, -1))

        if self._hp.action_space_type == 'discrete':
            target_actions = make_one_hot(inputs.action[:, 0, 0].long(), self._hp.n_action).type(torch.float)
            losses.update(AttrDict({'action_'+name: BCEWithLogitsLoss()(model_output.actions[name][inputs.task_id==i],
                                                                        target_actions[inputs.task_id==i])
                                    for i, name in enumerate(self._task_names)}))
        elif self._hp.action_space_type == 'continuous':
            inputs.task_id = inputs.task_id.unsqueeze(1).repeat(1, inputs.action.shape[1]).reshape(-1)
            losses.update(AttrDict({'action_'+name: L2Loss()(model_output.actions[name][inputs.task_id==i],
                                                             inputs.action.view(inputs.action.shape[0]*inputs.action.shape[1],*inputs.action.shape[2:])[inputs.task_id==i])
                                    for i, name in enumerate(self._task_names)}))
        else:
            raise NotImplementedError

        # print('** losses',losses)
        losses.total = self._compute_total_loss(losses)
        return losses
