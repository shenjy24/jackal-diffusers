from functools import partial

import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image, StableDiffusionUpscalePipeline


def paint_image(request_id, positive_prompt, negative_prompt, upscale_model=None):
    """
    加载本地模型进行AI绘图
    :param request_id: 请求ID，唯一标识，如果不为空则触发进度回调
    :param positive_prompt: 正面提示词
    :param negative_prompt: 负面提示词
    :param upscale_model: 放大模型，为空则不放大
    """
    origin_img = generate_image(request_id, positive_prompt, negative_prompt)
    if not upscale_model:
        return origin_img
    # 提升图片清晰度
    upscale_img = upscale_image(request_id, upscale_model, positive_prompt, negative_prompt, origin_img)
    return upscale_img


def generate_image(request_id, positive_prompt, negative_prompt):
    """
    AI生成图片
    :param request_id: 请求ID，唯一标识，如果不为空则触发进度回调
    :param positive_prompt: 正面提示词
    :param negative_prompt: 负面提示词
    """

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    adapter_id = "latent-consistency/lcm-lora-sdxl"

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16",
                                                     local_files_only=True)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()

    # load and fuse lcm lora
    pipe.load_lora_weights(adapter_id, local_files_only=True)
    pipe.fuse_lora()

    # disable guidance_scale by passing 0
    callback = partial(progress, request_id) if request_id else None
    image = pipe(prompt=positive_prompt, negative_prompt=negative_prompt, num_inference_steps=4,
                 guidance_scale=0, width=512, height=512, callback=callback).images[0]

    image.save(f"image.png")
    return image


def upscale_image(request_id, model, positive_prompt, negative_prompt, low_res_img):
    """
    放大图片，提高图片分辨率
    :param request_id: 请求ID
    :param model model: 模型名
    :param positive_prompt: 正面提示词
    :param negative_prompt: 负面提示词
    :param low_res_img: 低分辨率图片
    :return:
    """
    # load model and scheduler
    # model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(model, local_files_only=True, safety_checker=None)
    pipeline = pipeline.to("cuda")
    callback = partial(progress, request_id) if request_id else None
    upscale_img = pipeline(prompt=positive_prompt, negative_prompt=negative_prompt, image=low_res_img,
                           num_inference_steps=20, callback=callback).images[0]
    return upscale_img


def latent_upscale_image(request_id, model, positive_prompt, negative_prompt, low_res_img):
    """
    提升图片清晰度
    :param request_id: 请求ID，唯一标识，如果不为空则触发进度回调
    :param model: 模型
    :param positive_prompt: 正面提示词
    :param negative_prompt: 负面提示词
    :param low_res_img: 需要放大的图片
    """
    # load model and scheduler
    # model = "stabilityai/sd-x2-latent-upscaler"
    pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(
        model, torch_dtype=torch.float16
    )
    pipeline = pipeline.to(get_device())
    generator = torch.manual_seed(33)
    callback = partial(progress, request_id) if request_id else None
    upscale_img = pipeline(prompt=positive_prompt, negative_prompt=negative_prompt, image=low_res_img,
                           num_inference_steps=20, callback=callback, guidance_scale=0, generator=generator).images[0]
    return upscale_img


def progress(request_id, step, timestep, latents):
    """
    绘图过程的回调方法，用于展示进度
    :param request_id:     请求ID
    :param step:    步骤
    :param timestep:
    :param latents:
    :return:
    """
    content = {"request_id": request_id, "step": step}
    # redis_publisher.publish_message("image_step", json.dumps(content))


def get_device():
    if torch.cuda.is_available():
        print("cuda is available")
        return "cuda"
    else:
        print("cuda is not available")
        return "cpu"


if __name__ == '__main__':
    print(torch.cuda.is_available())
    pp = "Best quality, masterpiece, ultra high res, (photorealistic:1.4), 1girl"
    np = "ng_deepnegative_v1_75t, badhandv4"
    img = paint_image(request_id=None, positive_prompt=pp, negative_prompt=np,
                      upscale_model="stabilityai/stable-diffusion-x4-upscaler")
    img.save(f"imgage.png")
