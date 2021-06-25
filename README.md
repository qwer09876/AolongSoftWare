### 详细源码

本项目使用了Metric Learning模型和deepSort_paddle进行开发，基于[寂寞你快进去](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/180581)提供的行人追踪开源项目。

- 行人重识别（Person Re-IDentification，简称Re-ID）,也称行人再识别，是利用计算机视觉技术判断图像或者视频序列中是否存在特定行人的技术
- 广泛被认为是一个图像检索的子问题。给定一个监控行人图像，检索跨设备下的该行人图像
- 旨在弥补目前固定的摄像头的视觉局限，并可与行人检测/行人跟踪技术相结合，可广泛应用于智能视频监控、智能安保等领域
- 具体任务为：一个区域有多个摄像头拍摄视频序列，ReID的要求对一个摄像头下感兴趣的行人，检索到该行人在其他摄像头下出现的所有图片

- 本项目基于Paddle官方模型库中的[Metric Learning](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/metric_learning)模型开发


###使用的开源代码为deepsort相关的：
##使用开源代码文件结构如下；
#deep_sort   存放deepsort的核心算法
#model       存放模型
#mainTest.py 使用进行对视频的测试



### **数据集介绍**与处理

- 本次使用的数据集是经典的行人重识别数据集Market-1501
- Market-1501 数据集在清华大学校园中采集，夏天拍摄，在 2015 年构建并公开
- 它包括由6个摄像头（其中5个高清摄像头和1个低清摄像头）拍摄到的 1501 个行人、32668 个检测到的行人矩形框
- 每个行人至少由2个摄像头捕获到，并且在一个摄像头中可能具有多张图像
- 训练集有 751 人，包含 12,936 张图像，平均每个人有 17.2 张训练数据；测试集有 750 人，包含 19,732 张图像，平均每个人有 26.3 张测试数据
- 3368 张查询图像的行人检测矩形框是人工绘制的，而 gallery 中的行人检测矩形框则是使用DPM检测器检测得到的
- 该数据集提供的固定数量的训练集和测试集均可以在single-shot或multi-shot测试设置下使用

```python
#解压数据集
%cd ~/work/metric_learning/data
!tar -xf ~/data/data1884/Market-1501-v15.09.15.tar
!mv Market-1501-v15.09.15 Market-1501

#数据集预处理
%cd ~
import os
import random
imgs = os.listdir('work/metric_learning/data/Market-1501/gt_bbox')
imgs.sort()
image_id = 1
super_class_id = 1
class_id = 1
class_dict = {}
for img in imgs:
    if 'jpg' in img:
        person_id = img[:4]
        if person_id not in class_dict:
            class_dict[person_id] = class_id
            class_id += 1

total = []
for img in imgs:
    if 'jpg' in img:
        person_id = img[:4]
        path = 'gt_bbox/'+img
        line = '%s %s %s %s\n' % (image_id, class_dict[person_id], super_class_id, path)
        image_id += 1
        total.append(line)

# random.shuffle(total)
train = total[:-1014]
dev = total[-1014:]


with open('work/metric_learning/data/Market-1501/train.txt', 'w', encoding='UTF-8') as f:
    f.write('image_id class_id super_class_id path\n')
    for line in train:
        f.write(line)
    
with open('work/metric_learning/data/Market-1501/test.txt', 'w', encoding='UTF-8') as f:
    f.write('image_id class_id super_class_id path\n')
    for line in dev:
        f.write(line)

```



### 下载预训练模型

- 选择的BackBone是ResNet101_vd

  ```python
  %cd ~/work/metric_learning/pretrained_model
  
  !wget https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_ssld_pretrained.tar
  
  !tar -xf ResNet101_vd_ssld_pretrained.tar
  ```

  

# **模型训练**

- 使用softmax loss或者arcmargin loss进行模型训练

```python
%cd ~/work/metric_learning

!python train_elem.py \
    --model ResNet101_vd \
    --embedding_size 128 \
    --train_batch_size 256 \
    --test_batch_size 256 \
    --image_shape 3,128,128 \
    --class_dim 1421 \
    --lr 0.001 \
    --lr_strategy piecewise_decay \
    --lr_steps 5000,7000,9000 \
    --total_iter_num 12000 \
    --display_iter_step 10 \
    --test_iter_step 500 \
    --save_iter_step 500 \
    --use_gpu True \
    --pretrained_model pretrained_model/ResNet101_vd_ssld_pretrained \
    --model_save_dir save_elem101_model \
    --loss_name arcmargin \
    --arc_scale 80.0 \
    --arc_margin 0.15 \
    --arc_easy_margin False


```

### 使用eml

```python

#模型使用eml微调，达到最优效果


%cd ~/work/metric_learning
#跑到了4500轮
!python train_pair.py \
    --model ResNet101_vd \
    --embedding_size 64 \
    --train_batch_size 64 \
    --test_batch_size 64 \
    --image_shape 3,128,128 \
    --class_dim 1421 \
    --lr 0.001 \
    --lr_strategy piecewise_decay \
    --lr_steps 3000,6000,9000 \
    --total_iter_num 12000 \
    --display_iter_step 10 \
    --test_iter_step 500 \
    --save_iter_step 500 \
    --use_gpu True \
    --pretrained_model save_pair_model/ResNet101_vd/4500 \
    --model_save_dir save_pair_model \
    --loss_name eml \
    --samples_each_class 4 \
    --margin 0.1 \
    --npairs_reg_lambda 0.01
```



### 模型导出

```python
%cd ~/work/metric_learning

!python export_model.py \
    --model ResNet101_vd \
    --embedding_size 128 \
    --image_shape 3,128,128 \
    --use_gpu True \
    --pretrained_model=save_pair_model/ResNet101_vd/5500 \
    --model_save_dir=save_inference_model

        #     --pretrained_model=save_elem101_model/ResNet101_vd/12000 \

```



得到相应的模型后使用DeepSort进行追踪



### **DeepSort**

- DeepSort是在Sort目标追踪基础上的改进
- 引入了在行人重识别数据集上离线训练的深度学习模型，在实时目标追踪过程中，提取目标的表观特征进行最近邻匹配，可以改善有遮挡情况下的目标追踪效果
- 同时，也减少了目标ID跳变的问题
- DeepSort算法的大致流程如下图：

![img](https://ai-studio-static-online.cdn.bcebos.com/09635b7b0084477d9652ca1f77c9bedee52509ce178b4ce693f6941ec2f62246)

```python
# 克隆代码库
!git clone https://github.com/jm12138/deep_sort_paddle 
    
    
# 解压预训练模型
!unzip data/data59950/model.zip -d deep_sort_paddle

# 使用测试视频进行预测推理
%cd ~/deep_sort_paddle

!python main.py \
    --video_path ~/data/data59950/PETS09-S2L1-raw.webm \
    --save_dir output \
    --threshold 0.1 \
    --use_gpu
```


###软件部分
#软件部分使用的是基于Flask框架的web项目
文件目录介绍
#static     存放各种后端需要资源，css,js,img,video等
#templates  存放前端界面文件
#app.py     项目启动py
#config.py  项目的配置文件
#predict.py 对前端数据进行后端的处理，并与deepSort的算法进行一系列的融合处理
#videoWriterTest.py  测试当前的视频合成，是否可在浏览器中显示。
