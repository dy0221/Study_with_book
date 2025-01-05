import sys, os
# 내경로가 현재 폴더가 아닌 상위 폴더로 바꿈 # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) 
import cv2
import numpy as np
import pickle
from collections import OrderedDict
from Backpropagation.relu_layer import Relu
from Backpropagation.affine_layer import Affine
from cnn.convolution_layer import Convolution
from cnn.polling_layer import Pooling

# OpenCV 버전 확인
print('====================================')
print("Python version:", sys.version)  
print("NumPy version:", np.__version__) 
print("OpenCV version:", cv2.__version__) 
print('====================================')


class InferenceModel:
    '''
    mode 0 << image
    mode 1 << cam
    '''
    def __init__(self, mode, image_file, pkl_file): 

        input_dim = (1, 28, 28)
        conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}
        hidden_size=100 
        output_size=10 
        weight_init_std=0.01
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        if (mode == 0):
            self.image = self.import_image(image_file)
            self.process_image = self.preprocess_image(self.image)
        else:
            self.cam = self.import_cam()

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        self.load_params(pkl_file)

    def predict(self, x):
        for layer in self.layers.values():
            # print(f"Layer: {layer.__class__.__name__}")
            x = layer.forward(x)
        return x
    
    def inference(self, y):
        predicted_class = np.argmax(y)  # 가장 큰 값의 인덱스를 구함
        return predicted_class
        
    def import_image(self, image_file):
        image = cv2.imread(image_file)
        
        if image is None:
            raise Exception("can't read image")
        
        return image
    
    def import_cam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("웹캠을 열 수 없습니다.")
        
        return cap
    
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
        resized = cv2.resize(gray, (28, 28))  # 28x28 크기로 조정
        normalized = resized / 255.0  # 정규화
        reshaped = normalized.reshape(1, 1, 28, 28)  # input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
        return reshaped
    
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]

    def show_resized_image(self, row_image, max_width=800):
        # 이미지의 원본 크기
        height, width = row_image.shape[:2]
        
        # 화면에 맞는 최대 너비를 설정하고, 비율에 맞춰 높이 계산
        if width > max_width:
            aspect_ratio = height / width
            new_width = max_width
            new_height = int(new_width * aspect_ratio)
            resized_row_image = cv2.resize(row_image, (new_width, new_height))
        else:
            resized_row_image = row_image
        return resized_row_image
    

if __name__=="__main__":
    image = 'dataset\\number_example.png'
    pkl_file = 'params.pkl'
    mode = 1
    inference_model = InferenceModel(mode, image, pkl_file)
    if (mode == 0):
        y = inference_model.predict(inference_model.process_image)
        y = inference_model.inference(y)
        print('====================================')
        print('\n\n추론 값 : ', y, '\n\n')
        print('====================================')

        cv2.imshow("Image", inference_model.show_resized_image(inference_model.image))  
        cv2.waitKey(0)  
        cv2.destroyAllWindows()  
    else:
        while True:
            ret, frame = inference_model.cam.read() 
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            cv2.imshow('Webcam', frame)

            y = inference_model.predict(inference_model.preprocess_image(frame))
            y = inference_model.inference(y)
            print(f"\r추론 값 : {y} ", end="", flush=True)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        inference_model.cam.release()
        cv2.destroyAllWindows()
