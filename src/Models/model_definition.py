
import numpy as np
import torch
import timm

class GeM(torch.nn.Module):
    
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
    
    def gem(self,x, p=3, eps=1e-6):
        return torch.nn.functional.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class HCD_Simple_Model(torch.nn.Module):

    def __init__(self, model_name, num_classes = 1,**kwargs):
        super(HCD_Simple_Model, self).__init__()

        self.name = model_name
        
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=3)
        self.GeM_pooling = GeM()
        # Capa completamente conectada (fully connected)
        self.fc = torch.nn.Linear(16 , num_classes)  # Ajustar según tamaño de la imagen
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):

        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.nn.functional.relu(self.conv2(x))
        x = self.GeM_pooling(x).flatten(1)

        x = self.fc(x)
        x = self.sigmoid(x)

        return x

class HCD_Model_ResNet(torch.nn.Module):

    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None, freeze_params = True, device = torch.device('cuda')):
        super(HCD_Model_ResNet, self).__init__()
        model = timm.create_model(model_name=model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.model = model.to(device)

        in_features = self.model.fc.in_features

        self.model.fc = torch.nn.Identity()
        self.model.global_pool = torch.nn.Identity()

        if freeze_params:
            for param in self.model.parameters():
                param.requires_grad = False

        self.pooling = GeM()
        self.linear = torch.nn.Linear(in_features, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
  
        return output
    
class HCD_Model_EVA02(torch.nn.Module):

    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None, freeze_params = True, out_size=384, device = torch.device('cuda')):
        super(HCD_Model_EVA02, self).__init__()
        model = timm.create_model(model_name=model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.model = model.to(device)

        if freeze_params:
            for param in self.model.parameters():
                param.requires_grad = False

        self.linear = torch.nn.Linear(out_size, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        output = self.sigmoid(self.linear(features))
        return output

class HCD_Model_EffNet(torch.nn.Module):

    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None, freeze_params = True, device = torch.device('cuda')):
        super(HCD_Model_EffNet, self).__init__()
        model = timm.create_model(model_name=model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.model = model.to(device)

        in_features = self.model.classifier.in_features

        self.model.classifier = torch.nn.Identity()
        self.model.global_pool = torch.nn.Identity()

        if freeze_params:
            for param in self.model.parameters():
                param.requires_grad = False

        self.pooling = GeM()
        self.linear = torch.nn.Linear(in_features, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))

        return output

class HCD_Model_ConvNext(torch.nn.Module):

    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None, freeze_params = True, device = torch.device('cuda')):
        super(HCD_Model_ConvNext, self).__init__()
        model = timm.create_model(model_name=model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.model = model.to(device)

        in_features = self.model.head.in_features

        self.model.head = torch.nn.Identity()

        if freeze_params:
            for param in self.model.parameters():
                param.requires_grad = False

        self.pooling = GeM()
        self.linear = torch.nn.Linear(in_features, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
        return output

class HCD_Model_Swin(torch.nn.Module):

    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None, freeze_params = True, device = torch.device('cuda')):
        super(HCD_Model_Swin, self).__init__()
        model = timm.create_model(model_name=model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.model = model.to(device)

        in_features = self.model.head.in_features

        self.model.head = torch.nn.Identity()

        if freeze_params:
            for param in self.model.parameters():
                param.requires_grad = False

        self.pooling = GeM()
        self.linear = torch.nn.Linear(in_features, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, images):
        features = self.model(images).permute(0,3,1,2)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
        return output
    
class HCD_Model_ViT(torch.nn.Module):

    def __init__(self, model_name, num_classes=1, pretrained=True, checkpoint_path=None, freeze_params = True, device = torch.device('cuda')):
        super(HCD_Model_ViT, self).__init__()
        model = timm.create_model(model_name=model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
        self.model = model.to(device)

        in_features = self.model.head.in_features

        self.model.head = torch.nn.Identity()

        if freeze_params:
            for param in self.model.parameters():
                param.requires_grad = False

        self.linear = torch.nn.Linear(in_features, num_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, images):
        features = self.model(images)
        output = self.sigmoid(self.linear(features))
        return output

# Diccionario para mapear configuración y clases
maping_model = {"HCD_Model_ResNet" : HCD_Model_ResNet,
               "HCD_Model_EVA02"  : HCD_Model_EVA02,
               "HCD_Model_EffNet" : HCD_Model_EffNet,
               "HCD_Simple_Model" : HCD_Simple_Model,
               "HCD_Model_ConvNext": HCD_Model_ConvNext,
               "HCD_Model_Swin": HCD_Model_Swin,
               "HCD_Model_ViT": HCD_Model_ViT}