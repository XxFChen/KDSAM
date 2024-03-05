import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedFeatureFusionAdapter(nn.Module):
    def __init__(self, input_channels, output_channels, reduction=16):
        super(EnhancedFeatureFusionAdapter, self).__init__()
        # Define 1x1 convolutions for channel adjustment
        self.adjust_input_channels = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        # Other layers remain unchanged
        self.dynamic_weight_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_channels * 2, output_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(output_channels // reduction, output_channels, 1),
            nn.Sigmoid()
        )
        self.feature_enhance_layer = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.feature_complementary_enhancement = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        )
        self.adjust_layer = nn.Conv2d(output_channels, output_channels, 1)

def forward(self, feature_primary, feature_secondary, scale_factor_secondary=1.0):
    # Adjust the channel number of the secondary feature to match the primary feature
    feature_secondary_adjusted = self.adjust_input_channels(feature_secondary)
    # Resize the secondary feature to match the primary feature's size
    if scale_factor_secondary != 1.0:
        feature_secondary_adjusted = F.interpolate(feature_secondary_adjusted, scale_factor=scale_factor_secondary, mode='bilinear', align_corners=True)
    # Concatenate the adjusted features
    concat_feature = torch.cat([feature_primary, feature_secondary_adjusted], dim=1)
    # Generate fusion weights and apply them
    dynamic_weights = self.dynamic_weight_layer(concat_feature)
    weighted_feature = dynamic_weights * concat_feature
    
    feature_primary_m = feature_primary * weighted_feature
    feature_secondary_m = feature_secondary * weighted_feature
    feature_fuse = feature_primary_m + feature_secondary_m
    
    # Process the weighted feature
    enhanced_feature = self.feature_enhance_layer(feature_fuse)
    final_feature = self.feature_complementary_enhancement(enhanced_feature)
    final_adjusted_feature = self.adjust_layer(final_feature)
    
    return final_adjusted_feature
