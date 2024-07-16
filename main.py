import torch
from diffusers import StableDiffusionXLPipeline
from prompts import sample_prompt
import schduler
from models import local_model_path
import time


def get_file_name() -> str:
    timestamp = int(time.time())
    return f"image_{timestamp}.png"


if __name__ == '__main__':
    # 모델 로드
    pipe = StableDiffusionXLPipeline.from_single_file(local_model_path, torch_dtype=torch.float16)
    pipe.scheduler = schduler.dpm_2m_sde_karras_scheduler
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    # 이미지 생성
    prompt = sample_prompt
    negative = ""
    step, guidance = (20, 0.75)
    image = pipe(
        prompt,
        guidance_scale=guidance,
        num_inference_steps=step,
        negative_prompt=negative,
    ).images[0]

    # 이미지 저장
    image.save(f"./output/{get_file_name()}")
