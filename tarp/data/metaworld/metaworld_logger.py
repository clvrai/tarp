import numpy as np
import os
import io
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET
import mujoco_py
from matplotlib import pyplot as plt
import pandas as pd
import six

from tarp.utils.general_utils import batchwise_index, split_along_axis
from tarp.components.logger import Logger
from tarp.data.metaworld.src.metaworld_utils import MetaWorldEnvHandler, DEFAULT_PIXEL_RENDER_KWARGS
from tarp.models.skill_space_mdl import SkillSpaceLogger
from tarp.models.rpl_mdl import RPLLogger
from tarp.utils.vis_utils import add_caption_to_img


class MetaworldLogger(Logger):
    # logger to save visualizations of input and output trajectories in metaworld environment

    @staticmethod
    def _init_env_from_id(id):
        return MetaWorldEnvHandler.env_from_id(id)

    @staticmethod
    def _render_state(env, model_xml, obs, name=""):
        # parse the xml string
        tree = ET.ElementTree(ET.fromstring(model_xml))
        root = tree.getroot()

        # get directories to mesh and textures
        asset_dir = os.path.join(os.environ['DATA_DIR'], 'metaworld/assets')
        mesh_dir = os.path.join(asset_dir, 'meshes')
        textures_dir = os.path.join(asset_dir, 'textures')

        # modify mesh and texture directory
        compiler_ele = root.find(".//compiler")
        compiler_ele.set("meshdir", mesh_dir)
        compiler_ele.set("texturedir", textures_dir)

        # all lights cast shadow
        for light in root.findall(".//light"):
            light.set("castshadow", "true")

        # modify camera position and angle
        camera_ele = root.find(".//camera")
        camera_ele.set("name", "headview")
        camera_ele.set("pos", "0 -0.7 0.75")
        camera_ele.set("quat", "0.7 0.5 0 0")

        # delete several sections from the xml file
        mujoco_ele = root.find(".")
        mujoco_ele.remove(root.find(".//keyframe"))
        mujoco_ele.remove(root.find(".//equality"))
        mujoco_ele.remove(root.find(".//actuator"))

        # remove the robot
        worldbody_ele = root.find(".//worldbody")
        worldbody_ele.remove(root.find(".//body[@name='base']"))

        # add first point
        geom1 = ET.SubElement(worldbody_ele, "geom")
        geom1.set("pos", "{} {} {}".format(*obs[:3]))
        geom1.set("size", "0.01 0.01 0.01")
        geom1.set("rgba", "0.259 0.522 0.957 1")  # blue, 4285f4
        geom1.text = None

        # add second point
        geom2 = ET.SubElement(worldbody_ele, "geom")
        geom2.set("pos", "{} {} {}".format(*obs[3:]))
        geom2.set("size", "0.01 0.01 0.01")
        geom2.set("rgba", "0.984 0.737 0.02 1")  # yellow, fbbc05
        geom2.text = None

        with NamedTemporaryFile(delete=False, suffix='.xml') as f:
            # write processed xml to file
            f.write(ET.tostring(root))
            filename = f.name

        # load the mujoco model from the updated xml file (now with annotations)
        model = mujoco_py.load_model_from_path(filename)
        os.remove(filename)

        # render the image of the model
        width, height = DEFAULT_PIXEL_RENDER_KWARGS['width'], DEFAULT_PIXEL_RENDER_KWARGS['height']
        sim = mujoco_py.MjSim(model)
        viewer = mujoco_py.MjRenderContextOffscreen(sim, 0)
        viewer.render(width, height, 0)
        img = np.asarray(viewer.read_pixels(width, height, depth=False)[::-1, :, :], dtype=np.uint8)

        # free memory
        del viewer

        # add caption to the image
        info = {
            "Environment": env.__class__.__name__,
            "Pt 1 (blue)": np.round(obs[:3], 2),
            "Pt 2 (yellow)": np.round(obs[3:], 2)
        }
        img = add_caption_to_img(img, info, name, flip_rgb=True)

        return img


class RPLMetaWorldLogger(MetaworldLogger, RPLLogger):
    def visualize(self, model_output, inputs, losses, step, phase, logger):
        """Visualizes RPL model outputs in MetaWorld environments."""
        self._log_loss_breakdown(inputs, losses, step, phase)
        RPLLogger.visualize(self, model_output, inputs, losses, step, phase, logger)

    def _log_loss_breakdown(self, inputs, losses, step, phase):
        """Log loss breakdown by tasks to image."""
        num_envs = MetaWorldEnvHandler.num_envs()
        env_ids = inputs.env_id.detach().cpu().tolist()
        hl_mse_breakdown = losses.hl_mse.breakdown.detach().cpu().tolist()
        ll_mse_breakdown = losses.ll_mse.breakdown.detach().cpu().tolist()

        env_ids_onehot = np.eye(num_envs)[env_ids]
        hl_mse_sum = env_ids_onehot.T.dot(np.reshape(hl_mse_breakdown, (env_ids_onehot.shape[0], 1))).flatten().round(2)
        ll_mse_sum = env_ids_onehot.T.dot(np.reshape(ll_mse_breakdown, (env_ids_onehot.shape[0], 1))).flatten().round(2)
        count = env_ids_onehot.sum(0).astype(int)

        df = pd.DataFrame()
        df['id'] = list(range(num_envs))
        df[''] = [''] * num_envs # empty column to leave space for environment names
        df['env_name'] = [MetaWorldEnvHandler.id2name(i) for i in range(num_envs)]
        df['hl_mse'] = np.round(hl_mse_sum / count, 3)
        df['ll_mse'] = np.round(ll_mse_sum / count, 3)
        df['count'] = count

        def render_mpl_table(data, col_width=2.0, row_height=0.625, font_size=14,
                             header_color='#4285f4', row_colors=['#f1f1f2', 'w'], edge_color='w',
                             bbox=[0, 0, 1, 1], header_columns=0,
                             ax=None, **kwargs):
            if ax is None:
                size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
                fig, ax = plt.subplots(figsize=size)
                ax.axis('off')

            mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(font_size)

            for k, cell in six.iteritems(mpl_table._cells):
                cell.set_edgecolor(edge_color)
                if k[0] == 0 or k[1] < header_columns:
                    cell.set_text_props(weight='bold', color='w')
                    cell.set_facecolor(header_color)
                else:
                    cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
            return ax

        fig = plt.figure(figsize=(14, 20))
        ax = fig.add_subplot(111)
        render_mpl_table(df, header_columns=0, col_width=2.5, ax=ax)
        ax.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = img.astype(float) / 255
        plt.close(fig)

        self.log_images(np.array([img]), "loss_breakdown", step, phase)


class SkillSpaceMetaWorldLogger(MetaworldLogger, SkillSpaceLogger):
    pass
