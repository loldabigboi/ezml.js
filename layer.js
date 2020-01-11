class FullyConnectedLayer {

    constructor(numNeurons, numInputs, options) {

        if (!numNeurons) {
            throw new Error("Number of neurons not specified.");
        } else if (!numInputs) {
            throw new Error("Number of inputs for next layer not specified.");
        }

        this.numInputs  = numInputs;  // can be deduced during compile() for hidden layers excl. first one
        this.numNeurons = numNeurons;
        
        this.activationType = (options.activationType) ? options.activationType : 
                                                         LayerConstants.SIGMOID;
        if (this.activationType == LayerConstants.SIGMOID) {
            this.activationFn        = sigmoid;
            this.dActivationFn       = dSigmoid;
            this.gDFn                = oneToOneActivationGD;
        } else if (this.activationType == LayerConstants.TANH) {
            this.activationFn        = tanh;
            this.dActivationFn       = dTanh;
            this.gDFn                = oneToOneActivationGD;
        } else if (this.activationType == LayerConstants.RELU) {
            this.activationFn        = relu;
            this.dActivationFn       = dRelu;
            this.gDFn                = oneToOneActivationGD;
        } else if (this.activationType == LayerConstants.SOFTMAX) {
            if (this.numNeuronsNext) {  // this is not output layer
                throw new Error("Softmax can only be used for output layer atm.");
            }
            this.activationFn        = softmax;
            this.gDFn                = crossEntropySoftmaxGD;
        }

        this.neurons = new Matrix(numNeurons, 1);

        this.biases = new Matrix(numNeurons, 1);
        this.minBias = (options.minBias) ? options.minBias : -1;
        this.maxBias = (options.maxBias) ? options.maxBias :  1;
        this.biases.map(() => {
            return this.minBias + Math.random()*(this.maxBias - this.minBias);
        });    

        this.weights = new Matrix(numNeurons, numInputs);
        this.minWeight = (options.minWeight) ? options.minWeight : -1;
        this.maxWeight = (options.maxWeight) ? options.maxWeight :  1;
        this.weights.map(() => {
            return this.minWeight + Math.random()*(this.maxWeight - this.minWeight);
        });

    }

}

class LayerConstants {}

// layer type constants
LayerConstants.FULLY_CONNECTED = "fully-connected";
LayerConstants.CONVOLUTIONAL = "convolutional";

// activation constants
LayerConstants.SIGMOID = "sigmoid";
LayerConstants.TANH = "tanh";
LayerConstants.RELU = "relu";
LayerConstants.SOFTMAX = "softmax";

// gd function types
LayerConstants.SOFTMAX_ENTROPY = "softmax-entropy";
LayerConstants.ONE_TO_ONE      = "one-to-one";  // *

// *
// use when the activation of a neuron is only dependent upon a single logit,
// e.g sigmoid, tanh or relu. e.g. dont use for softmax, as the activation of a
// neuron is dependent upon all logits for that layer.