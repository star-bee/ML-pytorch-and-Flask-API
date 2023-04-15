import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sqlite3

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png', 'gif'])


class vision_model(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(vision_model, self).__init__()
        self.tmodel = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
        for param in self.tmodel.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.tmodel(x))
        x = self.fc1(x)
        return x


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = torch.load(r'assets\model.pt', map_location=torch.device('cpu'))
model.eval()

def predict(image, device):
    clasess = ['airplane',  'automobile',  'bird',  'cat',  'deer',  'dog',  'frog',  'horse',  'ship',  'truck' ]
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, 1)
    return clasess[pred]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file and allowed_file(file.filename):
            # save the uploaded file
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img = Image.open(file)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            img_tensor = transform(img).unsqueeze(0)
            pred_y = predict(img_tensor, device)

            conn = sqlite3.connect('delameta.db')
            c = conn.cursor()
            c.execute("INSERT INTO prediction_result (filename, prediction) VALUES (?, ?)", (filename, pred_y))
            conn.commit()
            conn.close()

            return jsonify({
                'filename': filename,
                'content_type': file.content_type,
                'class': pred_y,
                'size': os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            })

    # return a simple upload form for GET requests
    return '''
    <!doctype html>
    <html>
    <body>
        <h1>Upload an Image</h1>
        <form method=post enctype=multipart/form-data>
            <input type=file name=file>
            <input type=submit value=Upload>
        </form>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(debug=True, host="0.0.0.0", port=8000)
