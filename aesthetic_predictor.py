import torch
from torch import nn
from transformers import AutoProcessor, CLIPVisionModelWithProjection


model_cache_ = dict()


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
    def __init__(
        self, model_path, clip_model="openai/clip-vit-large-patch14", device="default"
    ):
        self.device = device
        if self.device == "default":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load clip
        if clip_model in model_cache_:
            self.clip_model, self.transform = model_cache_[clip_model]
        else:
            self.clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model).to(
                self.device
            )
            self.transform = AutoProcessor.from_pretrained(clip_model)
            model_cache_[clip_model] = (self.clip_model, self.transform)
        dim = self.clip_model.config.projection_dim
        # load model
        self.model = MLP(dim)
        state_dict = torch.load(model_path)
        state_dict = state_dict.get("state_dict", state_dict)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, images=None, embeds=None):
        """
        Predict aesthetic scores.

        Args (images or embeds but not both):
            images: (optional) iterable of PIL Image
            embeds: (optional) embeddings returned by the get_embeds method
        Returns:
            list of aesthetic scores
        """
        if (images is None and embeds is None) or (images is not None and embeds is not None):
            raise ValueError("exactly one of images or embeds required")

        if images is not None:
            embeds = self.get_embeds(images)
        with torch.inference_mode():
            prediction = self.model(embeds)
        return prediction.squeeze(dim=-1).tolist()

    def get_embeds(self, images):
        """
        Get CLIP embeddings for a set of images. Useful for passing to multiple models.

        Args:
            images: iterable of PIL Image
        Returns:
            torch.Tensor of CLIP embeddings
        """
        images = torch.vstack(
            [
                self.transform(images=img, return_tensors="pt")["pixel_values"].to(
                    self.device
                )
                for img in images
            ]
        )
        with torch.inference_mode():
            embeds = self.clip_model(images).image_embeds
        return embeds
