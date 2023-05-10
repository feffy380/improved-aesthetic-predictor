import clip
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticPredictor:
    def __init__(self, model_path, clip_model="ViT-L/14", device="default"):
        self.device = device
        if self.device == "default":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load clip
        self.clip_model, self.transform = clip.load(clip_model, device=self.device)
        dim = self.clip_model.visual.output_dim
        # load model
        self.model = MLP(dim)
        state_dict = torch.load(model_path)
        state_dict = state_dict.get("state_dict", state_dict)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images):
        """
        Predict aesthetic scores.

        Args:
            images: iterable of PIL Image
        Returns:
            list of aesthetic scores
        """
        images = torch.vstack(
            [self.transform(img).unsqueeze(0).to(self.device) for img in images]
        )
        with torch.no_grad():
            features = self.clip_model.encode_image(images).type(torch.float)
            prediction = self.model(features)
        return prediction.squeeze(-1).tolist()
