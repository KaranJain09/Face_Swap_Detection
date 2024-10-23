from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model once at startup
model_path = os.path.join(os.getcwd(), 'myapp/models', 'fake_real_model.keras')
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully!")
else:
    raise FileNotFoundError(f"Model file not found: {model_path}")

def home(request):
    return render(request, 'index.html')

# Helper function to preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(128, 128))  # Resize image to 128x128
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@csrf_exempt
def analyze_image_view(request):
    if request.method == 'POST':
        image = request.FILES['image']

        # Save the uploaded image temporarily
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, image.name)

        with open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        try:
            # Preprocess and predict
            img_array = preprocess_image(image_path)
            prediction = model.predict(img_array)[0][0]  # Get the prediction value

            # Determine the result (0 = Fake, 1 = Real)
            result = 1 if prediction > 0.5 else 0

        finally:
            # Clean up by deleting the temporary image
            os.remove(image_path)

        # Send the result back as JSON
        return JsonResponse({'result': result})

    return JsonResponse({'error': 'Invalid request'}, status=400)
