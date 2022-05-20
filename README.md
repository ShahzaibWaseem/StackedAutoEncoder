# Stacked Autoencoder

## Model Description
This stacked autoencoder has three denoising autoencoders where their inputs are carried forward to the next autoencoder with the use of concatination.
This Concatination operation is applied to the input to (any of) the autoencoder with the output of the previous autoencoder.
Noise operation, if caried out (`noise` flag) is done on the concatenated input to the autoencoder, so the noise is not applied twice to one half of the input (input of the previous autoencoder).
If the feature maps are strided upon in the convolution layer, the carry forward stream (to the next autoencoder) is also convolved upon.

The bottleneck layers of the three autoencoders are flattened out and concatenated to eachother and the classification layer is attached to the end of this layer.
If the `individual_training` is set, the flattened bottleneck layers of the individual autoencoders are trained with the final Fully Connected layer, and then the flattened layer is concatenated to eachother for overall training of the whole system.

### Flags
The `noise` flag adds noise to the input, after the input of the previous autoencoder is concatenated to the output of the previous autoencoder. This makes the autoencoders **Deniosing Autoencoders**.
The Individual Autoencoders can be trained individually with the use of a flag `individual_training` and the whole system is trained as a whole as well.
Setting the `verbose_printing` prints out the individual loss values of the three autoencoders.

## Model Architecture
In the Convolutions (Autoencoder and carry forward block) the number written on the top tells how the channels of the feature maps develop. Whereas, the number written on the bottom is the spatial size.

![Model Architecture](model.svg)

## Configuration Tested on
- Python 3.8.7
- Pytorch 1.8.1 (with Cuda 11.1)