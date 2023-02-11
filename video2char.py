import os
import re
import shutil
from tqdm import trange, tqdm
import cv2
from PIL import Image, ImageFont, ImageDraw
from moviepy.editor import VideoFileClip


class V2Char:
    font_path = "Arial.ttf"
    ascii_char = "#8XOHLTI)i=+;:,. "

    def __init__(self, video_path, clarity):
        self.video_path = video_path
        self.clarity = clarity

    def video2str(self):
        def convert(img):
            if img.shape[0] > self.text_size[1] or img.shape[1] > self.text_size[0]:
                img = cv2.resize(img, self.text_size, interpolation=cv2.INTER_NEAREST)
            ascii_frame = ""
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    ascii_frame += self.ascii_char[
                        int(img[i, j] / 256 * len(self.ascii_char))
                    ]
            return ascii_frame

        print("正在将原视频转为字符...")
        self.char_video = []
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        font_size = int(25 - 20 * max(min(float(self.clarity), 1), 0))
        self.font = ImageFont.truetype(self.font_path, font_size)
        self.char_width, self.char_height = max(
            [self.font.getsize(c) for c in self.ascii_char]
        )
        self.text_size = (
            int(self.raw_width / self.char_width),
            int(self.raw_height / self.char_height),
        )
        for _ in trange(self.nframe):
            raw_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
            frame = convert(raw_frame)
            self.char_video.append(frame)
        cap.release()

    def str2fig(self):
        print("正在生成字符图像...")
        col, row = self.text_size
        catalog = self.video_path.split(".")[0]
        if not os.path.exists(catalog):
            os.makedirs(catalog)
        blank_width = int((self.raw_width - self.text_size[0] * self.char_width) / 2)
        blank_height = int((self.raw_height - self.text_size[1] * self.char_height) / 2)
        for p_id in trange(len(self.char_video)):
            strs = [self.char_video[p_id][i * col : (i + 1) * col] for i in range(row)]
            im = Image.new("RGB", (self.raw_width, self.raw_height), (255, 255, 255))
            dr = ImageDraw.Draw(im)
            for i, str in enumerate(strs):
                for j in range(len(str)):
                    dr.text(
                        (
                            blank_width + j * self.char_width,
                            blank_height + i * self.char_height,
                        ),
                        str[j],
                        font=self.font,
                        fill="#000000",
                    )
            im.save(catalog + r"/pic_{}.jpg".format(p_id))

    def jpg2video(self):
        print("正在将字符图像合成字符视频...")
        catalog = self.video_path.split(".")[0]
        images = os.listdir(catalog)
        images.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
        im = Image.open(catalog + "/" + images[0])
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        savedname = catalog.split("/")[-1]
        vw = cv2.VideoWriter(savedname + "_tmp.mp4", fourcc, self.fps, im.size)
        for image in tqdm(images):
            frame = cv2.imread(catalog + "/" + image)
            vw.write(frame)
        vw.release()
        shutil.rmtree(catalog)

    def merge_audio(self):
        print("正在将音频合成到字符视频中...")
        raw_video = VideoFileClip(self.video_path)
        char_video = VideoFileClip(self.video_path.split(".")[0] + "_tmp.mp4")
        audio = raw_video.audio
        video = char_video.set_audio(audio)
        video.write_videofile(
            self.video_path.split(".")[0] + f"_{self.clarity}.mp4",
            codec="libx264",
            audio_codec="aac",
        )
        os.remove(self.video_path.split(".")[0] + "_tmp.mp4")

    def gen_video(self):
        self.video2str()
        self.str2fig()
        self.jpg2video()
        self.merge_audio()


if __name__ == "__main__":
    video_path = input("输入视频文件路径:\n")
    clarity = input("输入清晰度(0~1, 直接回车使用默认值0):\n") or 0
    v2char = V2Char(video_path, clarity)
    v2char.gen_video()
