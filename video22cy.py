import os
from tqdm import trange
import wget
import cv2
from PIL import Image
from moviepy.editor import VideoFileClip
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from deepdanbooru.model import DeepDanbooruModel
from deepdanbooru.util import tag


class V22cy:
    def __init__(self, video_path, model_id="cag/anything-v3-1"):
        self.video_path = video_path
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
        self.pipe.to("cuda:0")
        self.tag_model = DeepDanbooruModel()
        if not os.path.exists("model-resnet_custom_v3.pt"):
            wget.download(
                "https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt"
            )
        self.tag_model.load_state_dict(
            torch.load("model-resnet_custom_v3.pt", map_location="cpu")
        )
        self.tag_model.eval()
        self.tag_model.half()
        self.tag_model.to("cuda:0")

    def read_video(self):
        print("正在读取视频信息并抽帧...")
        cap = cv2.VideoCapture(self.video_path)
        self.nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if not os.path.exists("frames"):
            os.mkdir("frames")
        for idx in trange(self.nframe):
            frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).convert("RGB")
            img.save(f"frames/{idx}.png")
        cap.release()

    def img2img(self):
        print("正在生成二次元图像...")
        if not os.path.exists("frames-2cy"):
            os.mkdir("frames-2cy")
        for idx in trange(self.nframe):
            init_img = Image.open(f"frames/{idx}.png").convert("RGB")
            prompt = tag(init_img, self.tag_model)
            negative_prompt = "Low Quality, Bad Anatomy"
            img = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_img,
                num_inference_steps=20,
                strength=0.5,
                guidance_scale=7,
                eta=0.0,
            ).images[0]
            img.save(f"frames-2cy/{idx}.png")

    def img2video(self):
        print("正在将二次元图像合成二次元视频...")
        im = Image.open(f"frames-2cy/0.png")
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        vw = cv2.VideoWriter("tmp.mp4", fourcc, self.fps, im.size)
        for idx in trange(self.nframe):
            img_url = f"frames-2cy/{idx}.png"
            frame = cv2.imread(img_url)
            vw.write(frame)
        vw.release()

    def merge_audio(self):
        print("正在将音频合成到字符视频中...")
        raw_video = VideoFileClip(self.video_path)
        tmp_video = VideoFileClip("tmp.mp4")
        audio = raw_video.audio
        video = tmp_video.set_audio(audio)
        filename = self.video_path.split("/")[-1].split(".")[0].strip()
        video.write_videofile(
            f"{filename}_2cy.mp4",
            codec="libx264",
            audio_codec="aac",
        )
        os.remove("tmp.mp4")

    def gen_video(self):
        self.read_video()
        self.img2img()
        self.img2video()
        self.merge_audio()


if __name__ == "__main__":
    video_path = input("输入视频文件路径:\n")
    model_id = "cag/anything-v3-1"
    v22cy = V22cy(video_path, model_id=model_id)
    v22cy.gen_video()
