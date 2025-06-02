from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions
from io import BytesIO
import os
import base64

def predict(request):
    if request.method == 'GET':
        return render(request, 'home.html', {
            'form': ImageUploadForm(),
            'img_data': None,
            'top5': None,
            'show_result': False
        })

    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        img_file = form.cleaned_data['image']
        img_bytes = img_file.read()
        img_stream = BytesIO(img_bytes)

        # VGG16が要求する画像サイズにリサイズ
        img = load_img(img_stream, target_size=(224, 224))
        img_array = img_to_array(img).reshape(1, 224, 224, 3) / 255.0

        model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'model.h5')
        model = load_model(model_path)
        preds = model.predict(img_array)

        top5 = decode_predictions(preds, top=5)[0]

        img_data = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"

        return render(request, 'home.html', {
            'form': form,
            'img_data': img_data,
            'top5': top5,
            'show_result': True
        })

    # フォームが無効な場合も HttpResponse を返す
    return render(request, 'home.html', {
        'form': form,
        'img_data': None,
        'top5': None,
        'show_result': False
    })
