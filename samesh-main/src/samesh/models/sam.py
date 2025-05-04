import re
from pathlib import Path
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoModel

USE_SAMHQ = False
if USE_SAMHQ:
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
else:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


from samesh.data.common import NumpyTensor

try:
    import folder_paths # ComfyUI specific
except ImportError:
    # Provide a fallback or raise a more informative error if needed outside ComfyUI
    print("Warning: folder_paths module not found. SAM model path resolution might fail.")
    folder_paths = None

try:
    from omegaconf import OmegaConf
except ImportError:
    print("Warning: OmegaConf not found. Config loading will fail.")
    OmegaConf = None


def combine_bmasks(masks: NumpyTensor['n h w'], sort=False) -> NumpyTensor['h w']:
    """
    """
    mask_combined = np.zeros_like(masks[0], dtype=int)
    if sort:
        masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    for i, mask in enumerate(masks):
        mask_combined[mask] = i + 1
    return mask_combined


def decompose_mask(mask: NumpyTensor['h w'], background=0) -> NumpyTensor['n h w']:
    """
    """
    labels = np.unique(mask)
    labels = labels[labels != background]
    return mask == labels[:, None, None]


def remove_artifacts(mask: NumpyTensor['h w'], mode: str, min_area=128) -> NumpyTensor['h w']:
    """
    Removes small islands/fill holes from a mask.
    """
    assert mode in ['holes', 'islands']
    mode_holes = (mode == 'holes')

    def remove_helper(bmask):
        # opencv connected components operates on binary masks only
        bmask = (mode_holes ^ bmask).astype(np.uint8)
        nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
        sizes = stats[:, -1][1:]  # Row 0 corresponds to 0 pixels
        fill = [i + 1 for i, s in enumerate(sizes) if s < min_area] + [0]
        if not mode_holes:
            fill = [i for i in range(nregions) if i not in fill]
        return np.isin(regions, fill)

    mask_combined = np.zeros_like(mask)
    for label in np.unique(mask): # also process background
        mask_combined[remove_helper(mask == label)] = label
    return mask_combined


def colormap_mask(
    mask : NumpyTensor['h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), foreground=None, blend=0.25
) -> Image.Image:
    """
    """
    palette = np.random.randint(0, 255, (np.max(mask) + 1, 3))
    palette[0] = background
    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground
    image_mask = palette[mask.astype(int)] # type conversion for boolean masks
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)


def colormap_bmask(bmask: NumpyTensor['h w']) -> Image.Image:
    """
    """
    return colormap_mask(bmask, background=np.array([0, 0, 0]), foreground=np.array([255, 255, 255]))


def colormap_bmasks(
    masks: NumpyTensor['n h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), blend=0.25
) -> Image.Image:
    """
    """
    mask = combine_bmasks(masks)
    return colormap_mask(mask, image, background=background, blend=blend)


def point_grid_from_mask(mask: NumpyTensor['h w'], n: int) -> NumpyTensor['n 2']:
    """
    Sample points within valid mask normalized to [0, 1] x [0, 1]
    """
    valid = np.argwhere(mask)
    if len(valid) == 0:
        raise ValueError('No valid points in mask')

    h, w = mask.shape
    n = min(n, len(valid))
    indices = np.random.choice(len(valid), n, replace=False)
    samples = valid[indices].astype(float)
    samples[:, 0] /= h - 1
    samples[:, 1] /= w - 1
    samples = samples[:, [1, 0]]
    samples = samples[np.lexsort((samples[:, 1], samples[:, 0]))]
    return samples


class SamModel(nn.Module):
    """
    """
    def __init__(self, config: OmegaConf, device='cuda'):
        """
        """
        super().__init__()
        self.config = config
        self.device = device
        
        if config.sam.auto:
            self.setup_sam(mode='auto')
        else:
            if config.sam.ground:
                self.setup_grounding_dino()
            self.setup_sam(mode='pred')

    def setup_sam(self, mode='auto'):
        """
        """
        match = re.search(r'vit_(l|tiny|h)', self.config.sam.checkpoint)
        self.sam_model = sam_model_registry[match.group(0)](checkpoint=self.config.sam.checkpoint)
        self.sam_model = self.sam_model.to(self.device)
        self.sam_model.eval()
        self.engine = {
            'pred': SamPredictor,
            'auto': SamAutomaticMaskGenerator,
        }[mode](self.sam_model, **self.config.sam.get('engine_config', {}))

    def setup_grounding_dino(self):
        """
        """
        self.grounding_dino_processor, self.grounding_dino_model = \
            AutoProcessor.from_pretrained(self.config.grounding_dino.checkpoint), \
            AutoModel    .from_pretrained(self.config.grounding_dino.checkpoint).to(self.device)

    def process_image(self, image: Image, prompt: dict = None) -> NumpyTensor['n h w']:
        """
        For information on prompt format see:
        
        https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py#L104
        """
        image = np.array(image)

        if self.config.sam.auto:
            annotations = self.engine.generate(image)
        else:
            self.engine.set_image(image)
            annotations = self.engine.predict(**prompt)[0]        
            annotations = [{'segmentation': m, 'area': m.sum().item()} for m in annotations]
        
        annotations = sorted(annotations, key=lambda x: x['area'], reverse=True)
        masks = np.stack([anno['segmentation'] for anno in annotations])
        return masks

    def process_boxes(self, image: Image, texts: list[str]) -> tuple[
        list[NumpyTensor[4]],
        list[NumpyTensor[2]]
    ]:
        """
        """
        texts = '. '.join(texts)
        inputs = self.grounding_dino_processor(texts, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.grounding_dino_model(**inputs)

        boxes, logits = self.grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
        )
        return boxes, logits

    def forward(self, image: Image, texts: list[str]=None) -> NumpyTensor['n h w']:
        """
        """
        if self.config.sam.auto:
            masks = self.process_image(image)
        else:
            boxes, _ = self.process_boxes(image, texts)
            masks = []
            for box in boxes:
                masks.append(self.process_image(image, {'box': box}))
            masks = np.concatenate(masks)
        return masks


class Sam2Model(SamModel):
    """
    """
    def setup_sam(self, mode='auto'):
        """
        """
        print('setting up sam model...')
        
        config_filename = self.config.sam.model_config # Expects filename like 'sam2_hiera_t.yaml'
        checkpoint_path = self.config.sam.checkpoint
        
        # --- Path Resolution Attempt 4: Hardcoded Base + Filename ---
        # Use the user-provided base path directly
        sam_models_dir = r"A:\ComfyUI+Sage2\ComfyUI\models\sam" # Raw string for windows path
        
        if not os.path.isdir(sam_models_dir):
             # Try to get ComfyUI base path as a last resort if hardcoded path fails
             if hasattr(folder_paths, 'base_path'):
                  sam_models_dir = os.path.join(folder_paths.base_path, "models", "sam")
                  print(f"Warning: Hardcoded SAM dir not found, trying Comfy base: {sam_models_dir}")
             else:
                  raise FileNotFoundError(f"Hardcoded SAM directory not found and could not get ComfyUI base path: {sam_models_dir}")

        full_config_path = os.path.join(sam_models_dir, config_filename)
        
        if not os.path.exists(full_config_path):
             raise FileNotFoundError(f"Could not find SAM config file at the expected path: {full_config_path}. Ensure '{config_filename}' exists in '{sam_models_dir}'.")
        
        print(f"SAM2 setup_sam: Using absolute config path: {full_config_path}")
        config_arg = full_config_path # Always use the resolved absolute path
        # --- End Path Resolution Attempt 4 ---

        # --- Model instantiation logic (remains the same, passes absolute path) ---
        if USE_SAMHQ and mode == 'auto':
             print(f"SAM2 setup_sam (USE_SAMHQ=True path): Calling build_sam2 with config: {config_arg}")
             self.sam_model = build_sam2(config_arg, checkpoint_path, device=self.device, apply_postprocessing=False)
        elif USE_SAMHQ and mode == 'ground':
             print(f"SAM2 setup_sam (USE_SAMHQ=True, mode=ground): Calling build_sam2_video_predictor with config: {config_arg}")
             self.sam_model = build_sam2_video_predictor(config_arg, checkpoint_path, device=self.device)
             self.sam_model = SamVideoPredictorHFWrapper(self.sam_model)
        else: # Default path (USE_SAMHQ=False or other modes)
             print(f"SAM2 setup_sam (Default path): Calling build_sam2 with config: {config_arg}")
             self.sam_model = build_sam2(config_arg, checkpoint_path, device=self.device, apply_postprocessing=False)
        
        self.sam_model.eval()
        
        engine_config = self.config.sam.get('engine_config', {})
        print(f"SAM2 setup_sam: Initializing engine (mode='{mode}') with config: {engine_config}")
        
        self.engine = {
            'pred': SAM2ImagePredictor,
            'auto': SAM2AutomaticMaskGenerator,
        }[mode](self.sam_model, **engine_config)


if __name__ == '__main__':
    import time

    device = 'cuda'
    image = Image.open('/home/ubuntu/meshseg/tests/examples/goldpot.png')

    config = OmegaConf.create({
        'sam': {
            'checkpoint': '/home/ubuntu/meshseg/checkpoints/sam_hq_vit_h.pth',
            'auto': True, 
            'ground': False,
            'engine_config': {'points_per_side': 32},
        },
        'grounding_dino': {
            'checkpoint': 'IDEA-Research/grounding-dino-tiny', # TODO find larger model
        },
    })

    sam = SamModel(config, device)
    start_time = time.time()
    masks = sam(image)
    print(f'Elapsed time: {time.time() - start_time:.2f} s')
    image = colormap_bmasks(masks, np.array(image))
    image.save('test_mask.png')

    '''
    config.sam.auto = False
    config.sam.ground = False
    sam = SamModel(config, device)
    masks = sam.process_image(image, prompt={
        'point_coords': np.array([[image.height // 2, image.width // 2]]),
        'point_labels': np.ones((1,)),
        'multimask_output': False
    })
    image = colormap_bmasks(masks, np.array(image))
    image.save('test_mask_grounded.png')
    '''

    # For RuntimeError: No available kernel. Aborting execution
    # https://github.com/facebookresearch/segment-anything-2/issues/48
    config2 = OmegaConf.create({
        'sam': {
            'model_config': 'sam2_hiera_l.yaml',
            'checkpoint'  : '/home/ubuntu/meshseg/checkpoints/sam2_hiera_large.pt',
            'auto': True, 
            'engine_config': {'points_per_side': 32},
        },
    })

    sam2 = Sam2Model(config2, device)
    start_time = time.time()
    masks = sam2(image)
    print(f'Elapsed time: {time.time() - start_time:.2f} s')
    image = colormap_bmasks(masks, np.array(image))
    image.save('test_mask2.png')