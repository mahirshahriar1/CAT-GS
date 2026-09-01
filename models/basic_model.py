import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, LateFusion

class AVClassifier(nn.Module):
    def __init__(self, role, modality, fusion_method, dataset):
        super(AVClassifier, self).__init__()

        self.role = role  # 'teacher' or 'student'
        self.modality = modality  # 'audio', 'visual', or 'multimodal'
        self.fusion_method = fusion_method
        self.dataset = dataset
        n_classes = {
            'VGGSound': 309,
            'KineticSound': 31,
            'CREMAD': 6,
            'AVE': 28,
            'AVMNIST': 10,
            'CGMNIST': 10
        }.get(self.dataset, None)

        if n_classes is None:
            raise NotImplementedError(f"Incorrect dataset name {config.dataset}")

        if self.modality in ['audio', 'visual']:
            if self.modality == 'visual' and config.dataset == 'AVMNIST':
                self.net = resnet18(modality='image')
            elif self.modality == 'audio' and config.dataset == 'CGMNIST':
                # For CGMNIST, modal1 (gray) uses 'image' modality (1 channel grayscale)
                self.net = resnet18(modality='image')
            elif self.modality == 'visual' and config.dataset == 'CGMNIST':
                # For CGMNIST, modal2 (colored) uses 'rgb_image' modality (3 channel RGB, single image)
                self.net = resnet18(modality='rgb_image')
            else:
                self.net = resnet18(modality=self.modality)
            self.classifier = nn.Linear(512, n_classes)  # Assuming output features of 512
            
 
        elif self.modality == 'multimodal':
            if self.fusion_method in ['sum', 'concat', 'film', 'gated', 'late']:
                fusion_class = {
                    'sum': SumFusion,
                    'concat': ConcatFusion,
                    'film': FiLM,
                    'gated': GatedFusion,
                    'late': LateFusion
                }[self.fusion_method]
                self.fusion_module = fusion_class(output_dim=n_classes)
            else:
                raise NotImplementedError(f"Incorrect fusion method: {self.fusion_method}")
            
            # For CGMNIST: modal1 (gray) uses 'image' modality, modal2 (colored) uses 'rgb_image' modality
            if config.dataset == 'CGMNIST':
                self.audio_net = resnet18(modality='image')  # Gray images (modal1, 1 channel)
                self.visual_net = resnet18(modality='rgb_image')  # Colored images (modal2, 3 channels, single image)
            elif config.dataset == 'AVMNIST':
                self.audio_net = resnet18(modality='audio')
                self.visual_net = resnet18(modality='image')
            else:
                self.audio_net = resnet18(modality='audio')
                self.visual_net = resnet18(modality='visual')
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")


    def forward(self, audio, visual, return_features=False):   
        if self.modality in ['audio', 'visual']:
            if self.modality == 'audio':
                input_data = audio
                features = self.net(input_data)
                features = F.adaptive_avg_pool2d(features, 1)
                features = torch.flatten(features, 1)
            
            elif self.modality == 'visual':
                input_data = visual
                features = self.net(input_data)
                
                # For CGMNIST visual (colored images), no temporal dimension
                if config.dataset == 'CGMNIST':
                    # Input is [batch, 3, 28, 28], features are [batch, C, H, W]
                    features = F.adaptive_avg_pool2d(features, 1)
                    features = torch.flatten(features, 1)
                else:
                    # For video datasets, reshape and pool the features
                    (_, C, H, W) = features.size()
                    B = input_data.size()[0]
                    features = features.view(B, -1, C, H, W)
                    features = features.permute(0, 2, 1, 3, 4)
                    if config.dataset == 'AVMNIST':
                        features = F.adaptive_avg_pool2d(features, 1)
                    else:
                        features = F.adaptive_avg_pool3d(features, 1)
                    features = torch.flatten(features, 1)
            
            if return_features:
                return features, self.classifier(features)
            else:
                return self.classifier(features)
        
        
        elif self.modality == 'multimodal':
            a_features = self.audio_net(audio)
            v_features = self.visual_net(visual)

            # Reshape and pool the features
            (_, C, H, W) = v_features.size()
            B = a_features.size()[0]
            
            # For CGMNIST, both modalities are single images (no temporal dimension)
            if config.dataset == 'CGMNIST':
                a_features = F.adaptive_avg_pool2d(a_features, 1)
                v_features = F.adaptive_avg_pool2d(v_features, 1)
            else:
                v_features = v_features.view(B, -1, C, H, W)
                v_features = v_features.permute(0, 2, 1, 3, 4)
                
                a_features = F.adaptive_avg_pool2d(a_features, 1)
                if config.dataset == 'AVMNIST':
                    v_features = F.adaptive_avg_pool2d(v_features, 1)
                else:
                    v_features = F.adaptive_avg_pool3d(v_features, 1)

            a_features = torch.flatten(a_features, 1)
            v_features = torch.flatten(v_features, 1)


            if return_features:
                a_output, v_output, fused_output = self.fusion_module(a_features, v_features)
                return a_output, v_output, fused_output
            
            else:
                _, _, fused_output = self.fusion_module(a_features, v_features)
                return fused_output
            
        else:
            raise ValueError("Invalid modality specified for the forward pass")



if __name__ == '__main__':
    model = AVClassifier('teacher', 'audio', 'concat', 'CREMAD')
    print(model)