from PIL import Image
import torch
from torchvision import transforms

import sys
# from models.birefnet import BiRefNet

import os
from glob import glob
from image_proc import refine_foreground

from utils import check_state_dict


from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)

# birefnet = BiRefNet(bb_pretrained=False)
# state_dict = torch.load('../BiRefNet-general-epoch_244.pth', map_location='cpu')
# state_dict = check_state_dict(state_dict)
# birefnet.load_state_dict(state_dict)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_float32_matmul_precision(['high', 'highest'][0])

birefnet.to(device)
birefnet.eval()
print('BiRefNet is ready to use.')

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




src_dir = sys.argv[1]
image_paths = glob(os.path.join(src_dir, '*'))
dst_dir = './predictions'
os.makedirs(dst_dir, exist_ok=True)
for image_path in image_paths:
    print('Processing {} ...'.format(image_path))
    image = Image.open(image_path)
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # Show Results
    pred_pil = transforms.ToPILImage()(pred)
    pred_pil.resize(image.size).save(image_path.replace(src_dir, dst_dir))

    # Visualize the last sample:
    # Scale proportionally with max length to 1024 for faster showing
    # scale_ratio = 1024 / max(image.size)
    # scaled_size = (int(image.size[0] * scale_ratio), int(image.size[1] * scale_ratio))

    # image_masked = refine_foreground(image, pred_pil)
    # image_masked.putalpha(pred_pil.resize(image.size))