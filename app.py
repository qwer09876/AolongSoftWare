from flask import Flask, render_template, request
import config
import json
import os

Threshold = 0
Confidence = 0

app = Flask(__name__)
app.config.from_object(config)


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/upload', methods=["GET", "POST"])
def upload():
    print("未选择图片")
    video = request.files['photo']
    global Threshold

    Threshold = request.form['Threshold']
    global Confidence
    Confidence = request.form['Confidence']
    print(video)
    if not video:
        print("未选择图片")
        return json.dumps(
            {
                'status': False,
                'change': '未选择图片'
            }
        )

    fn = video.filename
    print("文件名---->", fn)
    print('/static/upload/' + fn)
    video.save(os.path.join("static/upload/", fn))
    return json.dumps({
        'status': True,
        'change': "上传图片",
        'videoUrl': '/static/upload/' + fn
    })


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    videopath = request.form.get("videourl") + ''  # 获取post类型信息
    name = str(videopath).split("/")[-1]
    global Threshold
    global Confidence

    print(Confidence + "      ---------------           ", Threshold)
    from mainTest import SoftTest
    res = SoftTest(r'static/upload/' + name, name, Confidence ,Threshold)
    print("res----->", res)

    # path=r'/static/upload/eee.mp4'
    # print(path)
    # r'/static/upload/' + res
    return json.dumps({
        'status': True,
        'videoUrl': r'/static/output/' + res,
        'aaa': "static/log/log" + name.split(".")[0] + ".txt",
        'name': res.split(".")[0] + ".txt"
    })


#


if __name__ == '__main__':
    app.run(debug=True)
