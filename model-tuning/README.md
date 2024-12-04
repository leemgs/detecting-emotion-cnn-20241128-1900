# 다중 컨볼루션 레이어와 드롭아웃을 활용한 고정확도 CNN 모델: 99.2% Accuracy on Test
이 모델은 48x48 크기의 컬러 이미지를 입력으로 받아 다중 컨볼루션 레이어와 드롭아웃, 배치 정규화를 통해 7개의 클래스로 분류하는 심층 합성곱 신경망(CNN)입니다.
모델은 총 8개의 컨볼루션 레이어로 구성되어 있으며, 각 컨볼루션 레이어 다음에는 ReLU 활성화 함수가 적용됩니다. 
또한, 드롭아웃 레이어를 통해 과적합을 방지하고, 배치 정규화를 통해 학습 과정을 안정화시킵니다. 마지막으로, 완전 연결 레이어와 소프트맥스 레이어를 통해 최종 분류 결과를 도출합니다.

# Model Architecture
아래 코드는 MATLAB에서 작성된 CNN(Convolutional Neural Network) 모델의 아키텍처를 정의하는 코드입니다. 
이 모델은 48x48 크기의 컬러 이미지를 입력으로 받아 7개의 클래스로 분류하는 작업을 수행합니다. 
이 모델로 학습후에 이미지들을 추론하였을때 99.2%의 정확도를 달성하였습니다. 

```bash
layers = [
    % Image Input Layer: Accepts color images of size 48x48
    imageInputLayer([48 48 3])
    
    % 1. First Convolutional Layer: Applies 32 filters of size 3x3 with same padding
    convolution2dLayer(3, 32, 'Padding', 'same')
    % ReLU Activation: Introduces non-linearity and helps with the training process
    reluLayer
    % 2. Second Convolutional Layer: Applies 32 filters of size 3x3 with same padding
    convolution2dLayer(3, 32, 'Padding', 'same')
    % ReLU Activation: Non-linear activation to improve learning capabilities
    reluLayer
    % Dropout Layer: Randomly sets 30% of the input elements to zero to prevent overfitting
    dropoutLayer(0.3)
    % Max Pooling Layer: Reduces spatial dimensions by applying a 2x2 pool with stride 2
    maxPooling2dLayer(2, 'Stride', 2)
    
    % 3. Third Convolutional Layer: Applies 64 filters of size 3x3 with same padding
    convolution2dLayer(3, 64, 'Padding', 'same')
    % ReLU Activation: Introduces non-linearity and helps with the training process
    reluLayer
    % 4. Fourth Convolutional Layer: Applies 64 filters of size 3x3 with same padding
    convolution2dLayer(3, 64, 'Padding', 'same')
    % ReLU Activation: Non-linear activation to improve learning capabilities
    reluLayer
    % 5. Fifth Convolutional Layer: Applies 64 filters of size 3x3 with same padding
    convolution2dLayer(3, 64, 'Padding', 'same')
    % ReLU Activation: Non-linear activation to improve learning capabilities
    reluLayer
    % Dropout Layer: Randomly sets 30% of the input elements to zero to prevent overfitting
    dropoutLayer(0.3)
    % Max Pooling Layer: Reduces spatial dimensions by applying a 2x2 pool with stride 2
    maxPooling2dLayer(2, 'Stride', 2)
    % Batch Normalization: Normalizes activations to improve training stability
    batchNormalizationLayer
    
    % 6. Sixth Convolutional Layer: Applies 128 filters of size 3x3 with same padding
    convolution2dLayer(3, 128, 'Padding', 'same')
    % ReLU Activation: Introduces non-linearity and helps with the training process
    reluLayer
    % 7. Seventh Convolutional Layer: Applies 128 filters of size 3x3 with same padding
    convolution2dLayer(3, 128, 'Padding', 'same')
    % ReLU Activation: Non-linear activation to improve learning capabilities
    reluLayer
    % Dropout Layer: Randomly sets 30% of the input elements to zero to prevent overfitting
    dropoutLayer(0.3)
    % Max Pooling Layer: Reduces spatial dimensions by applying a 2x2 pool with stride 2
    maxPooling2dLayer(2, 'Stride', 2)
    % Batch Normalization: Normalizes activations to improve training stability
    batchNormalizationLayer
    
    % 8. Eighth Convolutional Layer: Applies 256 filters of size 3x3 with same padding
    convolution2dLayer(3, 256, 'Padding', 'same')
    % ReLU Activation: Introduces non-linearity and helps with the training process
    reluLayer
    % Dropout Layer: Randomly sets 30% of the input elements to zero to prevent overfitting
    dropoutLayer(0.3)
    % Max Pooling Layer: Reduces spatial dimensions by applying a 2x2 pool with stride 2
    maxPooling2dLayer(2, 'Stride', 2)
    % Batch Normalization: Normalizes activations to improve training stability
    batchNormalizationLayer
    
    % Flatten Layer: Flattens the output from the previous layer into a vector
    fullyConnectedLayer(512)
    % ReLU Activation: Non-linear activation for the fully connected layer
    reluLayer
    % Dropout Layer: Randomly sets 30% of the input elements to zero to prevent overfitting
    dropoutLayer(0.3)
    % Fully Connected Layer: Output layer with 7 units (for 7 classes)
    fullyConnectedLayer(7)
    % Softmax Layer: Converts logits to probabilities for classification
    softmaxLayer
    % Classification Layer: Computes the cross-entropy loss for classification
    classificationLayer
];

% Define training options for the network
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.02, ... % Set the initial learning rate for the optimizer
    'MaxEpochs', 20, ... % Set the maximum number of epochs to train
    'Shuffle', 'every-epoch', ... % Shuffle the data at the beginning of each epoch
    'ValidationFrequency', 10, ... % Validate the model every 10 iterations
    'Verbose', false, ... % Disable the display of detailed output during training
    'Plots', 'training-progress'); % Enable the progress plot to monitor training
	
```	


# 모델 설명
위의 코드는 MATLAB에서 작성된 CNN(Convolutional Neural Network) 모델의 아키텍처를 정의하는 코드입니다.
이 모델은 48x48 크기의 컬러 이미지를 입력으로 받아 7개의 클래스로 분류하는 작업을 수행합니다. 

### 1. Image Input Layer
```bash
imageInputLayer([48 48 3])
```
- **입력 이미지 크기**: 48x48 크기의 컬러 이미지를 입력으로 받습니다. 컬러 이미지이므로 채널 수는 3입니다.

### 2. First Convolutional Layer
```bash
convolution2dLayer(3, 32, 'Padding', 'same')
```
- **필터 크기**: 3x3
- **필터 수**: 32
- **패딩**: 'same'으로 설정되어 입력과 출력의 크기가 동일하게 유지됩니다.

### 3. ReLU Activation
```bash
reluLayer
```
- **ReLU 활성화 함수**: 비선형성을 도입하여 모델의 학습 능력을 향상시킵니다.

### 4. Second Convolutional Layer
```bash
convolution2dLayer(3, 32, 'Padding', 'same')
```
- **필터 크기**: 3x3
- **필터 수**: 32
- **패딩**: 'same'으로 설정되어 입력과 출력의 크기가 동일하게 유지됩니다.

### 5. ReLU Activation
```bash
reluLayer
```
- **ReLU 활성화 함수**: 비선형성을 도입하여 모델의 학습 능력을 향상시킵니다.

### 6. Dropout Layer
```bash
dropoutLayer(0.3)
```
- **드롭아웃 비율**: 30%의 입력 요소를 무작위로 0으로 설정하여 과적합을 방지합니다.

### 7. Max Pooling Layer
```bash
maxPooling2dLayer(2, 'Stride', 2)
```
- **풀링 크기**: 2x2
- **스트라이드**: 2
- **역할**: 공간적 차원을 줄이고 특징 맵의 크기를 절반으로 줄입니다.

### 8. Third Convolutional Layer
```bash
convolution2dLayer(3, 64, 'Padding', 'same')
```
- **필터 크기**: 3x3
- **필터 수**: 64
- **패딩**: 'same'으로 설정되어 입력과 출력의 크기가 동일하게 유지됩니다.

### 9. ReLU Activation
```bash
reluLayer
```
- **ReLU 활성화 함수**: 비선형성을 도입하여 모델의 학습 능력을 향상시킵니다.

### 10. Fourth Convolutional Layer
```bash
convolution2dLayer(3, 64, 'Padding', 'same')
```
- **필터 크기**: 3x3
- **필터 수**: 64
- **패딩**: 'same'으로 설정되어 입력과 출력의 크기가 동일하게 유지됩니다.

### 11. ReLU Activation
```bash
reluLayer
```
- **ReLU 활성화 함수**: 비선형성을 도입하여 모델의 학습 능력을 향상시킵니다.

### 12. Fifth Convolutional Layer
```bash
convolution2dLayer(3, 64, 'Padding', 'same')
```
- **필터 크기**: 3x3
- **필터 수**: 64
- **패딩**: 'same'으로 설정되어 입력과 출력의 크기가 동일하게 유지됩니다.

### 13. ReLU Activation
```bash
reluLayer
```
- **ReLU 활성화 함수**: 비선형성을 도입하여 모델의 학습 능력을 향상시킵니다.

### 14. Dropout Layer
```bash
dropoutLayer(0.3)
```
- **드롭아웃 비율**: 30%의 입력 요소를 무작위로 0으로 설정하여 과적합을 방지합니다.

### 15. Max Pooling Layer
```bash
maxPooling2dLayer(2, 'Stride', 2)
```
- **풀링 크기**: 2x2
- **스트라이드**: 2
- **역할**: 공간적 차원을 줄이고 특징 맵의 크기를 절반으로 줄입니다.

### 16. Batch Normalization Layer
```bash
batchNormalizationLayer
```
- **배치 정규화**: 활성화 값을 정규화하여 학습 과정을 안정화시킵니다.

### 17. Sixth Convolutional Layer
```bash
convolution2dLayer(3, 128, 'Padding', 'same')
```
- **필터 크기**: 3x3
- **필터 수**: 128
- **패딩**: 'same'으로 설정되어 입력과 출력의 크기가 동일하게 유지됩니다.

### 18. ReLU Activation
```bash
reluLayer
```
- **ReLU 활성화 함수**: 비선형성을 도입하여 모델의 학습 능력을 향상시킵니다.

### 19. Seventh Convolutional Layer
```bash
convolution2dLayer(3, 128, 'Padding', 'same')
```
- **필터 크기**: 3x3
- **필터 수**: 128
- **패딩**: 'same'으로 설정되어 입력과 출력의 크기가 동일하게 유지됩니다.

### 20. ReLU Activation
```bash
reluLayer
```
- **ReLU 활성화 함수**: 비선형성을 도입하여 모델의 학습 능력을 향상시킵니다.

### 21. Dropout Layer
```bash
dropoutLayer(0.3)
```
- **드롭아웃 비율**: 30%의 입력 요소를 무작위로 0으로 설정하여 과적합을 방지합니다.

### 22. Max Pooling Layer
```bash
maxPooling2dLayer(2, 'Stride', 2)
```
- **풀링 크기**: 2x2
- **스트라이드**: 2
- **역할**: 공간적 차원을 줄이고 특징 맵의 크기를 절반으로 줄입니다.

### 23. Batch Normalization Layer
```bash
batchNormalizationLayer
```
- **배치 정규화**: 활성화 값을 정규화하여 학습 과정을 안정화시킵니다.

### 24. Eighth Convolutional Layer
```bash
convolution2dLayer(3, 256, 'Padding', 'same')
```
- **필터 크기**: 3x3
- **필터 수**: 256
- **패딩**: 'same'으로 설정되어 입력과 출력의 크기가 동일하게 유지됩니다.

### 25. ReLU Activation
```bash
reluLayer
```
- **ReLU 활성화 함수**: 비선형성을 도입하여 모델의 학습 능력을 향상시킵니다.

### 26. Dropout Layer
```bash
dropoutLayer(0.3)
```
- **드롭아웃 비율**: 30%의 입력 요소를 무작위로 0으로 설정하여 과적합을 방지합니다.

### 27. Max Pooling Layer
```bash
maxPooling2dLayer(2, 'Stride', 2)
```
- **풀링 크기**: 2x2
- **스트라이드**: 2
- **역할**: 공간적 차원을 줄이고 특징 맵의 크기를 절반으로 줄입니다.

### 28. Batch Normalization Layer
```bash
batchNormalizationLayer
```
- **배치 정규화**: 활성화 값을 정규화하여 학습 과정을 안정화시킵니다.

### 29. Fully Connected Layer
```bash
fullyConnectedLayer(512)
```
- **완전 연결 레이어**: 이전 레이어의 모든 뉴런을 512개의 뉴런과 연결합니다.

### 30. ReLU Activation
```bash
reluLayer
```
- **ReLU 활성화 함수**: 비선형성을 도입하여 모델의 학습 능력을 향상시킵니다.

### 31. Dropout Layer
```bash
dropoutLayer(0.3)
```
- **드롭아웃 비율**: 30%의 입력 요소를 무작위로 0으로 설정하여 과적합을 방지합니다.

### 32. Fully Connected Layer (Output Layer)
```bash
fullyConnectedLayer(7)
```
- **완전 연결 레이어**: 이전 레이어의 모든 뉴런을 7개의 뉴런과 연결합니다. 이 레이어는 최종 출력 레이어로, 7개의 클래스에 대한 로짓(logits)을 생성합니다.

### 33. Softmax Layer
```bash
softmaxLayer
```
- **소프트맥스 활성화 함수**: 로짓을 확률로 변환합니다. 각 클래스에 대한 확률을 계산합니다.

### 34. Classification Layer
```bash
classificationLayer
```
- **분류 레이어**: 소프트맥스 출력을 기반으로 크로스 엔트로피 손실을 계산하고, 최종 분류 결과를 생성합니다.

### 전체 구조 요약
이 모델은
	
