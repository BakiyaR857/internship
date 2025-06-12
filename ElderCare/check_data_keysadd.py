import torch
import numpy as np
from torch.serialization import safe_globals

with safe_globals([np.core.multiarray._reconstruct]):
    try:
        data = torch.load("model/data.pth", weights_only=False)
        print("✅ File loaded successfully!")
        print("🔑 Available keys:", data.keys())
        print("📦 Full content:")
        print(data)
    except Exception as e:
        print("❌ Error loading file:", e)
