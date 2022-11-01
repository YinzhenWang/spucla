import os
import io
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import resnet_ym
# import timms
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import matplotlib.pyplot as plt
import faiss
import time
import numpy as np
import shutil

app = Flask(__name__)
CORS(app)  # 解决跨域问题

test_dir = "./all_images/"
weights_path = "./resnet34.pt"
class_json_path = "./class_indices.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."
model_name = "resnet"
num_classes = 152
batch_size = 32
input_size = 224
is_fixed = True
use_pretrained = True
topN = 4
unloader = transforms.ToPILImage()

def folder_clean():
    del_list = os.listdir("./static/images")
    for f in del_list:
        file_path = os.path.join("./static/images", f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    print("file is cleaned!")
            
def image_show(tensor,i):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.figure()
    plt.imshow(image)
    plt.savefig('./static/images/{}.jpg'.format(i))
    plt.close('all')
    print(i,"ok")

def get_datasets(data_dir, input_size, is_train_data):
    if (is_train_data):
        images = datasets.ImageFolder(train_dir,
                                      transforms.Compose([
                                          transforms.RandomResizedCrop(input_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()
                                      ]))
    else:
        images = datasets.ImageFolder(test_dir,
                                      transforms.Compose([
                                          transforms.Resize(input_size),
                                          transforms.CenterCrop(input_size),
                                          transforms.ToTensor()
                                      ]))
    return images

def cal(model, test_loader,target,imgs):
    topn = topN + 1
    simlist = [0 for i in range(topn)]
    imglist = [0 for i in range(topn)]
    model.eval()
    print("cal init")
    xb = np.fromfile("resnet34v512.bin",dtype="float64").astype("float32")
    xb.shape = 322045,512
    xq = np.array([target.cpu().tolist()]).astype("float32")
    xq = xq[0]
    nlist = 152
    d = 512
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer,d,nlist,faiss.METRIC_L2)
    gpu_index = faiss.index_cpu_to_all_gpus(index)
    print(gpu_index.is_trained)
    gpu_index.train(xb)
    print(gpu_index.is_trained)

    gpu_index.add(xb)
    start_time = time.time()*1000
    print("here")
    D,gt_nss = gpu_index.search(xq,5)
    print("here")
    print(gt_nss)
    end_time = time.time()*1000
    print("time:",end_time-start_time)
    
    rank = 0
    for img_index in gt_nss[0]:
        print(img_index)
        print(imgs[img_index])
        shutil.copyfile(imgs[img_index+1][0], './static/images/{}.jpg'.format(rank-1))
        rank += 1
        
        
    
def set_parameters_require_grad(model, is_fixed):
    # 默认parameter.requires_grad = True
    # 当采用固定预训练模型参数的方法进行训练时，将预训练模型的参数设置成不需要计算梯度
    if (is_fixed):
        for parameter in model.parameters():
            parameter.requires_grad = False

def init_model(model_name, num_classes, is_fixed, use_pretrained):
    if (model_name == 'resnet'):
        # 调用resnet模型，resnet18表示18层的resnet模型，
        # pretrained=True表示需要加载预训练好的模型参数，pretrained=False表示不加载预训练好的模型参数
        model = resnet_ym.resnet34(pretrained=use_pretrained)  # 调用预训练的resnet101模型
        # 设置参数是否需要计算梯度
        # is_fixed=True表示模型参数不需要跟新（不需要计算梯度）
        # is_fixed=False表示模型参数需要fineturn（需要计算梯度）
        set_parameters_require_grad(model, is_fixed)

        in_features = model.fc.in_features  # 取出全连接层的输入特征维度

        # 重新定义resnet18模型的全连接层,使其满足新的分类任务
        # 此时模型的全连接层默认需要计算梯度
        model.fc = nn.Linear(in_features, num_classes)

    return model

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes):
    try:
        folder_clean()
        tensor = transform_image(image_bytes=image_bytes)
        model.eval()
        with torch.no_grad():
            # target = model.forward(tensor).squeeze()
            target, last_featrues = model.forward(tensor)
            target = target.squeeze()
            outputs = torch.softmax(target, dim=0)
        cal(model, test_loader, last_featrues,test_images.imgs)
        print("here")
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        index_pre = index_pre[:7]
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create model
model = init_model(model_name, num_classes, is_fixed, use_pretrained)
model.load_state_dict(torch.load(weights_path))
model = model.to(device)
model.eval()
test_images = get_datasets(test_dir, input_size, is_train_data=False)
test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size)

# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)



@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)

