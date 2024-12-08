# High-accuracy CNN model with multiple convolutional layers and dropout: 99.2% Acceleration Test
The model is a deep convolutional neural network (CNN) that takes 48x48 size color images as input and classifies them into seven classes with multiple convolutional layers, dropout, and batch normalization.
The model consists of a total of eight convolutional layers, each of which is followed by a ReLU activation function. 
Furthermore, we prevent overfitting through dropout layers, and stabilize the learning process through batch normalization. Finally, we derive the final classification results through fully connected and softmax layers.

# Model Architecture
The code below defines the architecture of the Convolutional Neural Network (CNN) model written by MATLAB. 
```bash
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃     Param #   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Input_Layer (InputLayer)        │ (None, 48, 48, 1)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Conv1 (Conv2D)                  │ (None, 48, 48, 32)     │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ReLU1 (Activation)              │ (None, 48, 48, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Conv2 (Conv2D)                  │ (None, 48, 48, 32)     │         9,248 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ReLU2 (Activation)              │ (None, 48, 48, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout1 (Dropout)              │ (None, 48, 48, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ MaxPool1 (MaxPooling2D)         │ (None, 24, 24, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Conv3 (Conv2D)                  │ (None, 24, 24, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ReLU3 (Activation)              │ (None, 24, 24, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Conv4 (Conv2D)                  │ (None, 24, 24, 64)     │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ReLU4 (Activation)              │ (None, 24, 24, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Conv5 (Conv2D)                  │ (None, 24, 24, 64)     │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ReLU5 (Activation)              │ (None, 24, 24, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout2 (Dropout)              │ (None, 24, 24, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ MaxPool2 (MaxPooling2D)         │ (None, 12, 12, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ BatchNorm1 (BatchNormalization) │ (None, 12, 12, 64)     │           256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Conv6 (Conv2D)                  │ (None, 12, 12, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ReLU6 (Activation)              │ (None, 12, 12, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Conv7 (Conv2D)                  │ (None, 12, 12, 128)    │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ReLU7 (Activation)              │ (None, 12, 12, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout3 (Dropout)              │ (None, 12, 12, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ MaxPool3 (MaxPooling2D)         │ (None, 6, 6, 128)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ BatchNorm2 (BatchNormalization) │ (None, 6, 6, 128)      │           512 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Conv8 (Conv2D)                  │ (None, 6, 6, 256)      │       295,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ReLU8 (Activation)              │ (None, 6, 6, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout4 (Dropout)              │ (None, 6, 6, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ MaxPool4 (MaxPooling2D)         │ (None, 3, 3, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ BatchNorm3 (BatchNormalization) │ (None, 3, 3, 256)      │         1,024 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 2304)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ FC1 (Dense)                     │ (None, 512)            │     1,180,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ ReLU9 (Activation)              │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Dropout5 (Dropout)              │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ Output (Dense)                  │ (None, 7)              │         3,591 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,804,647 (6.88 MB)
 Trainable params: 1,803,751 (6.88 MB)
 Non-trainable params: 896 (3.50 KB)

```

The model takes 48x48 color images as inputs and classifies them into seven classes. 
When the images were inferred after training with this model, 99.2% accuracy was achieved.

```bash
layers = [
    % Image Input Layer: Accepts color images of size 48x48
    imageInputLayer([48 48 1])
    
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

# Model Description
The code above defines the architecture of the Convolutional Neural Network (CNN) model written by MATLAB.
The model takes 48x48 color images as inputs and classifies them into seven classes. 

#### 1. Image Input Layer
```bash
imageInputLayer([48 48 3])
```
- **Input image size**: Take a 48x48 color image as input. As it is a color image, the number of channels is 3.

#### 2. First Convolutional Layer
```bash
convolution2dLayer(3, 32, 'Padding', 'same')
```
- **Filter size**: 3x3
- **Number of filters**: 32
- **Padding**: Set to 'same', the input and output sizes remain the same.

#### 3. ReLU Activation
```bash
reluLayer
```
- **ReLU activation function**: Introduces nonlinearity to improve the model's learning capabilities.

#### 4. Second Convolutional Layer
```bash
convolution2dLayer(3, 32, 'Padding', 'same')
```
- **Filter size**: 3x3
- **Number of filters**: 32
- **Padding**: Set to 'same', the input and output sizes remain the same.

#### 5. ReLU Activation
```bash
reluLayer
```
- **ReLU activation function**: Introduces nonlinearity to improve the model's learning capabilities.

#### 6. Dropout Layer
```bash
dropoutLayer(0.3)
```
- **Dropout ratio**: Set 30% of inputs to 0 at random to prevent overfitting.

#### 7. Max Pooling Layer
```bash
maxPooling2dLayer(2, 'Stride', 2)
```
- **Pull size**: 2x2
- **Stride**: 2
- **Role**: Reduce spatial dimensions and halve the size of feature maps.

#### 8. Third Convolutional Layer
```bash
convolution2dLayer(3, 64, 'Padding', 'same')
```
- **Filter size**: 3x3
- **Number of filters**: 64
- **Padding**: Set to 'same', the input and output sizes remain the same.

#### 9. ReLU Activation
```bash
reluLayer
```
- **ReLU activation function**: Introduces nonlinearity to improve the model's learning capabilities.

#### 10. Fourth Convolutional Layer
```bash
convolution2dLayer(3, 64, 'Padding', 'same')
```
- **Filter size**: 3x3
- **Number of filters**: 64
- **Padding**: Set to 'same', the input and output sizes remain the same.

#### 11. ReLU Activation
```bash
reluLayer
```
- **ReLU activation function**: Introduces nonlinearity to improve the model's learning capabilities.

#### 12. Fifth Convolutional Layer
```bash
convolution2dLayer(3, 64, 'Padding', 'same')
```
- **Filter size**: 3x3
- **Number of filters**: 64
- **Padding**: Set to 'same', the input and output sizes remain the same.

#### 13. ReLU Activation
```bash
reluLayer
```
- **ReLU activation function**: Introduces nonlinearity to improve the model's learning capabilities.

#### 14. Dropout Layer
```bash
dropoutLayer(0.3)
```
- **Dropout ratio**: Set 30% of inputs to 0 at random to prevent overfitting.

#### 15. Max Pooling Layer
```bash
maxPooling2dLayer(2, 'Stride', 2)
```
- **Pull size**: 2x2
- **Stride**: 2
- **Role**: Reduce spatial dimensions and halve the size of feature maps.

#### 16. Batch Normalization Layer
```bash
batchNormalizationLayer
```
- **Batch Normalization**: Normalize activation values to stabilize the learning process.

#### 17. Sixth Convolutional Layer
```bash
convolution2dLayer(3, 128, 'Padding', 'same')
```
- **Filter size**: 3x3
- **Number of filters**: 128
- **Padding**: Set to 'same', the input and output sizes remain the same.

#### 18. ReLU Activation
```bash
reluLayer
```
- **ReLU activation function**: Introduces nonlinearity to improve the model's learning capabilities.

#### 19. Seventh Convolutional Layer
```bash
convolution2dLayer(3, 128, 'Padding', 'same')
```
- **Filter size**: 3x3
- **Number of filters**: 128
- **Padding**: Set to 'same', the input and output sizes remain the same.

#### 20. ReLU Activation
```bash
reluLayer
```
- **ReLU activation function**: Introduces nonlinearity to improve the model's learning capabilities.

#### 21. Dropout Layer
```bash
dropoutLayer(0.3)
```
- **Dropout ratio**: Set 30% of inputs to 0 at random to prevent overfitting.



End of line.
