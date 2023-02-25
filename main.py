from utils.predictions import create_prediction



if __name__ == "__main__":
    
    # image path
    img_path = 'data/inputs_for_train/test_upload/3095421.jpg'
    prediction = create_prediction(img_path)
    print(f'predicted class name: {prediction}')