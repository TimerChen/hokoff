# -*- coding: utf-8 -*-
import torch
import os


class LocalTorchPredictor(object):
    def __init__(self, net):
        super().__init__()
        self.device = torch.device("cpu")
        self.net = net.to(self.device)

    def load_model(self, model_path):
        model_filename = os.path.join(model_path, "model.pth")
        checkpoint = torch.load(model_filename, map_location=self.device)
        print("wabawaba")
        print(checkpoint.keys())
        for key in checkpoint["network_state_dict"]:
            if torch.isnan(checkpoint["network_state_dict"][key]).any():
                print("NaN values found in model parameters for key:", key)

        # 逐层检查模型权重
        for name, param in self.net.named_parameters():
            if torch.isnan(param).any():
                print("NaN values found in layer:", name)
        self.net.load_state_dict(checkpoint["network_state_dict"])

    def inference(self, data_list):
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in data_list]
        format_inputs = self.net.format_data(torch_inputs)
        self.net.eval()
        with torch.no_grad():
            rst_list = self.net(format_inputs)
        return rst_list
