# SkyPaint-Chinese-EN-v-1.0
#### SkyPaint是由奇点智源开发的中英双语文本生成图像的项目，目前还在持续更新优化中
- 项目地址: [SkyWorkAIGC-SkyPaint](https://github.com/SkyWorkAIGC/SkyPaint)

# 模型介绍
SkyPaint文本生成图片模型主要由两大部分组成，即提示词文本编码器模型和扩散模型两大部分。因此我们的优化也分为两步，首先基于[OpenAI-CLIP](https://github.com/openai/CLIP)优化了提示词文本编码器模型使得SkyPaint具有中英文识别能力，然后优化了扩散模型，使得SkyPaint具有现代艺术能力可以产生高质量图片。

# 模型功能
* 支持汉语和英文以及中英文混合提示词输入
* 支持生成现代艺术风格的高质量图片
* 支持stable_diffusion_1.x官方模型及相关微调模型的英文提示词
* 保留stable_diffusion提示词的使用习惯和方法

### SkyCLIP模型简介
SkyCLIP是我们采用一种高效的训练中英双语CLIP模型的方法得到的CLIP模型，该方法仅需要使用文本数据即可实现对[OpenAI-CLIP](https://github.com/openai/CLIP)模型的高效蒸馏，大幅降低了数据门槛，同时训练所需算力要求相较于原始CLIP模型减少90%以上，方便开源社区可以进行复现/微调。该方法仅改变了OpenAI-CLIP的文本编码器，可搭配使用OpenAI-CLIP的图像编码器实现图文检索功能。

### SkyCLIP训练数据来源
* 中英文机器翻译任务平行语料
* 联合国中英文平行语料
* [LAION](https://laion.ai/)中英文语料(部分)
* [Wukong](https://wukong-dataset.github.io/wukong-dataset/index.html)中文语料(部分)
* [AI-Challenger](https://github.com/AIChallenger)翻译任务中英文语料
* 古诗词中英文语料
* 提示词手册/魔法书中常见词组合而成的中英文语料

### SkyCLIP训练方法
将OpenAI-CLIP的text_encoder作为教师模型并且冻结参数，学生模型采用和教师模型同样大小的多语言BERT模型，训练时英文输入通过教师模型获取相应的t_en_hiddent_state，英文和中文分别通过学生模型获取相应s_en_hiddent_state，s_zh_hidden_state，采用l1、l2、cos距离等构造损失函数使得学生模型的中英文hiddent_state逐渐靠近教师模型的hiddent_state。由于平行语料的中文和英文存在天然的不等长性质，为了使得平行的中文和英文尽量接近，训练过程中我们还添加了中文解码器，使用学生模型的中英文hiddent_state作为解码器的hidden_state输入，通过翻译任务来辅助实现中文和英文的对齐目的。

### SkyCLIP模型评估
目前我们主要评估了SkyCLIP在[Flickr30K-CN](https://github.com/li-xirong/cross-lingual-cap)的zero-shot表现，主要对比了若干具备中文能力的相关开源模型，为确保对比的公平性，具有多个模型尺寸的我们均选取基于OpenAI-CLIP ViT-L/14尺寸的模型，我们评估的流程参考了[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)所提供的评估脚本。

**Flickr30K-CN Retrieval**:
<table border="1" width="150%">
	<tr align="center">
        <th>Task</th><th colspan="3">Text-to-Image</th><th colspan="3">Image-to-Text</th>
        <th rowspan="3">MR</th>
    </tr>
    <tr align="center">
        <th>Setup</th><th colspan="3">Zero-shot</th><th colspan="3">Zero-shot</th> 
    </tr>
    <tr align="center">
        <td>Metric</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td>
    </tr>
    <tr align="center">
        <td width="120%">Taiyi-326M</td><td>53.8</td><td>79.9</td><td>86.6</td><td>64.0</td><td>90.4</td><td>96.1</td><td>78.47</td>
    </tr>
    <tr align="center">
        <td width="120%">AltCLIP</td><td>50.7</td><td>75.4</td><td>83.1</td><td>73.4</td><td>92.8</td><td>96.9</td><td>78.72</td>
    </tr>
	<tr align="center">
        <td width="120%">Wukong</td><td>51.9</td><td>78.6</td><td>85.9</td><td>75</td><td>94.4</td><td>97.7</td><td>80.57</td>
    </tr>
	<tr align="center">
        <td width="120%">R2D2</td><td>42.6</td><td>69.5</td><td>78.6</td><td>63.0</td><td>90.1</td><td>96.4</td><td>73.37</td>
    </tr>
	<tr align="center">
        <td width="120%">CN-CLIP</td><td>68.1</td><td>89.7</td><td>94.5</td><td>80.2</td><td>96.6</td><td>98.2</td><td>87.87</td>
    </tr>
    <tr align="center">
        <td width="120%">SkyCLIP</td><td>58.8</td><td>82.6</td><td>89.6</td><td>78.8</td><td>96.1</td><td>98.3</td><td>84.04</td>
    </tr>
</table>
<br>

### SkyCLIP计算图文相似度
```py
from PIL import Image
import requests
import clip
import torch
from transformers import BertTokenizer
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel
import numpy as np

query_texts = ['一个人', '一辆汽车', '两个男人', '两个女人']  # 这里是输入提示词，可以随意替换。
# 加载SkyCLIP 中英文双语 text_encoder
text_tokenizer = BertTokenizer.from_pretrained("./tokenizer")
text_encoder = CLIPTextModel.from_pretrained("./text_encoder").eval()
text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']

url = "http://images.cocodataset.org/val2017/000000040083.jpg"  #这里可以换成任意图片的url
# 加载CLIP的image encoder
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_text_proj = clip_model.text_projection
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
image = processor(images=Image.open(requests.get(url, stream=True).raw), return_tensors="pt")

with torch.no_grad():
    image_features = clip_model.get_image_features(**image)
    text_features = text_encoder(text)[0]
    # sep_token对应于openai-clip的eot_token
    sep_index = torch.nonzero(text == student_tokenizer.sep_token_id)
    text_features = text_features[torch.arange(text.shape[0]), sep_index[:, 1]]
    # 乘text投影矩阵
    text_features = clip_text_proj(text_features)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # 计算余弦相似度 logit_scale是尺度系数
    logit_scale = clip_model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print(np.around(probs, 3))

```


### 扩散模型 Diffusion Model
我们的数据采用了筛选过的Laion数据集作为训练数据，同时在文本前面加上了 'sai-v1 art' 作为tag使模型能够更快速的学习到我们想要的风格及质量。
预训练模型采用了[stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 作为预训练，使用了16块A100训练了50个小时。
目前模型还在持续优化中，后续会有更稳定的模型更新

# 效果展示

### 中文
机械狗
![](results/1.png)

城堡 大海 夕阳 宫崎骏动画
![](results/2.png)

花落知多少
![](results/3.png)

半鸡半人，强壮
![](results/4.png)

鸡你太美
![](results/5.png)

## 测试用例

模型下载地址 [SkyPaint-v1.0](https://sai-hk.oss-cn-hongkong.aliyuncs.com/zb/skypaint-v-1.0.zip?OSSAccessKeyId=LTAI5tHuxqp63n5qw5eeB6Ji&Expires=1673528832&Signature=4PTeknRoXuHWmeQHXqgu8kB0q%2Bw%3D) 

```py
from diffusers import StableDiffusionPipeline

device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained("path_to_our_model").to(device)

prompts = [
    '机械狗',
    '城堡 大海 夕阳 宫崎骏动画',
    '花落知多少',
    '鸡你太美',
]

for prompt in prompts:
    prompt = 'sai-v1 art, ' + prompt
    image = pipe(prompt).images[0]  
    image.save("%s.jpg" % prompt)
```

# License
- [MIT License](LICENSE)
- [CreativeML Open RAIL-M](LICENSE-MODEL)
