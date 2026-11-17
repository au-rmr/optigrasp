# OptiGrasp affordance model: DINOv2 ViT backbone + DPT head with affordance, roll and pitch outputs.
# Original: Soofiyan Atar, Soofiyan/NBV-GP huggingface/Depth-Anything/depth_anything/dpt_angle_fine.py (unchanged since
# 82e2c19, 2024-05-21), itself built on Depth-Anything's dpt.py (Apache-2.0, see LICENSE.depth_anything).
# Changes here: imports; the backbone is built from the vendored DINOv2 code instead of torch.hub; unused
# imports, the Hugging Face mixin and the __main__ demo removed. Layer names are unchanged so the weights load as-is.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._dinov2 import vision_transformer as _vits
from .blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, rank=10, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class SelfAttention(nn.Module):
    """Implements a simple self-attention mechanism."""

    def __init__(self, in_channels, reduce_factor=4):
        super(SelfAttention, self).__init__()
        self.reduce_factor = reduce_factor
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Downsample
        x_downsampled = F.interpolate(
            x, scale_factor=1 / self.reduce_factor, mode="bilinear", align_corners=False
        )
        batch_size, C, height, width = x_downsampled.size()

        query = (
            self.query_conv(x_downsampled)
            .view(batch_size, -1, height * width)
            .permute(0, 2, 1)
        )
        key = self.key_conv(x_downsampled).view(batch_size, -1, height * width)
        value = (
            self.value_conv(x_downsampled)
            .view(batch_size, -1, height * width)
            .permute(0, 2, 1)
        )

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).view(batch_size, C, height, width)

        # Upsample to original size
        out_upsampled = F.interpolate(
            out, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )

        return out_upsampled + x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, stride=1
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, stride=1
        )
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class DPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
    ):
        super(DPTHead, self).__init__()

        self.nclass = nclass
        self.use_clstoken = use_clstoken
        self.combined_in_channels = in_channels + 0

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=self.combined_in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )
        
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn, rank=100)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn, rank=100)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, rank=100)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, rank=100)

        head_features_1 = features
        head_features_2 = 32

        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

        ######################################################################################################
        ######################################################################################################
        ######################################################################################################

        self.scratch.affordance_initial_conv = nn.Conv2d(
            head_features_1 + 1, head_features_1 // 2, kernel_size=3, padding=1
        )
        self.scratch.affordance_initial_bn = nn.BatchNorm2d(head_features_1 // 2)
        self.scratch.affordance_relu = nn.ReLU(inplace=True)

        # Adding simplified self-attention
        self.scratch.affordance_attention = SelfAttention(head_features_1 // 2)

        self.scratch.affordance_feature_extraction = nn.Sequential(
            nn.Conv2d(
                head_features_1 // 2,
                head_features_2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(head_features_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),  # Helps prevent overfitting
            nn.Conv2d(
                head_features_2, head_features_2 * 2, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(head_features_2 * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),  # Added another dropout layer
            # Adding additional layers
            nn.Conv2d(
                head_features_2 * 2,
                head_features_2 * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(head_features_2 * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                head_features_2 * 2, head_features_2, kernel_size=3, stride=1, padding=1
            ),  # Reducing channel size back
            nn.BatchNorm2d(head_features_2),
            nn.ReLU(inplace=True),
        )

        self.scratch.affordance_feature_extraction_extended = nn.Sequential(
            self.scratch.affordance_feature_extraction,
            ResidualBlock(head_features_2),
            ResidualBlock(head_features_2),
        )

        self.scratch.affordance_final_conv = nn.Conv2d(
            head_features_2, 3, kernel_size=1, stride=1
        )
        self.scratch.affordance_sigmoid = nn.Sigmoid()

    def forward(self, out_features, patch_h, patch_w, mask):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )
        out = self.scratch.output_conv2(out)

        seg_mask_resized = F.interpolate(
            mask.float(), size=path_1.shape[2:], mode="nearest"
        )

        # Concatenate the segmentation mask with patch_1 features
        patch_1_with_mask = torch.cat((path_1, seg_mask_resized), dim=1)

        x = self.scratch.affordance_initial_conv(patch_1_with_mask)
        x = self.scratch.affordance_initial_bn(x)
        x = self.scratch.affordance_relu(x)
        x = self.scratch.affordance_attention(x)  # Apply simplified self-attention here

        x = F.interpolate(
            x,
            size=(int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True,
        )

        x = self.scratch.affordance_feature_extraction_extended(x)
        # x = self.scratch.affordance_feature_extraction(x)

        x = self.scratch.affordance_final_conv(x)
        affordance_out = x[:, 0, :, :].unsqueeze(1)
        affordance_out_roll = x[:, 1, :, :].unsqueeze(1)
        affordance_out_pitch = x[:, 2, :, :].unsqueeze(1)
        affordance_out = self.scratch.affordance_sigmoid(affordance_out)
        affordance_out_roll = self.scratch.affordance_sigmoid(affordance_out_roll)
        affordance_out_pitch = self.scratch.affordance_sigmoid(affordance_out_pitch)

        affordance_out = affordance_out * mask.float()
        affordance_out_roll = affordance_out_roll * mask.float()
        affordance_out_pitch = affordance_out_pitch * mask.float()

        return out, affordance_out, affordance_out_roll, affordance_out_pitch


def _dinov2_backbone(encoder):
    """dinov2_{encoder}14 exactly as DINOv2's hubconf built it (pretrained=False: the weights file holds the backbone)."""
    arch = {"vits": "vit_small", "vitb": "vit_base", "vitl": "vit_large"}[encoder]
    return _vits.__dict__[arch](img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0)


class DPT_DINOv2(nn.Module):
    def __init__(
        self,
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        localhub=True,  # ignored: kept so configs that name it still work
    ):
        super(DPT_DINOv2, self).__init__()

        assert encoder in ["vits", "vitb", "vitl"]

        self.pretrained = _dinov2_backbone(encoder)

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.depth_head = DPTHead(
            1,
            dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
        )

    def forward(self, x, mask):

        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(
            x, 4, return_class_token=True
        )

        patch_h, patch_w = h // 14, w // 14

        depth, affordance_map, affordance_map_roll, affordance_map_pitch = self.depth_head(features, patch_h, patch_w, mask)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        affordance_map = F.interpolate(
            affordance_map, size=(h, w), mode="bilinear", align_corners=True
        )
        affordance_map = F.relu(affordance_map)

        affordance_map_roll = F.interpolate(
            affordance_map_roll, size=(h, w), mode="bilinear", align_corners=True
        )
        affordance_map_roll = F.relu(affordance_map_roll)

        affordance_map_pitch = F.interpolate(
            affordance_map_pitch, size=(h, w), mode="bilinear", align_corners=True
        )
        affordance_map_pitch = F.relu(affordance_map_pitch)


        return depth, affordance_map, affordance_map_roll, affordance_map_pitch


class DepthAnything(DPT_DINOv2):
    def __init__(self, config):
        super().__init__(**config)
