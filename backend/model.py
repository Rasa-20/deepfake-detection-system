import torch                 # PyTorch library for model loading
import torch.nn as nn        # nn for defining neural network layers
import torchvision           # to use pretrained video models

class VideoEncoder(nn.Module): # model learns visual features (movement, expressions, frame inconsistencies) using this block
    def __init__(self, out_features=512):
        super(VideoEncoder, self).__init__()
        self.model = torchvision.models.video.r2plus1d_18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_features)

    def forward(self, x):
        return self.model(x)

class AudioEncoder(nn.Module):  # block captures temporal/speech patterns like rhythm, tone shifts, or desyncs
    def __init__(self, input_size=40, hidden_size=128, num_layers=2):
        super(AudioEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, time_steps, n_mfcc]
        output, (hn, _) = self.lstm(x)
        return hn[-1]

class DeepfakeDetector(nn.Module):  # this is the deepfake detection model
    def __init__(self, video_feature_size=512, audio_feature_size=128, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        self.video_encoder = VideoEncoder(out_features=video_feature_size)
        self.audio_encoder = AudioEncoder(input_size=40, hidden_size=audio_feature_size)

        fusion_size = video_feature_size + audio_feature_size

        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, video, audio):
        video_feat = self.video_encoder(video)
        audio_feat = self.audio_encoder(audio)

        fused = torch.cat([video_feat, audio_feat], dim=1)
        return self.classifier(fused)
