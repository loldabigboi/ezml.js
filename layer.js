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

class ConvolutionalLayer {

    constructor(inputDimensions, options) {

        if (!inputDimensions) {
            throw new Error("Input dimensions not specified.");
        }
        this.inputDimensions = inputDimensions.slice();

        this.filterDimensions = options.filterDimensions ? options.filterDimensions : [3, 3];
        if (this.filterDimensions.length !== 2) {
            throw new Error("Filter can only be 2D");
        }

        this.numFilters = options.numFilters ? options.numFilters : 8;
        if (this.numFilters < 1) {
            throw new Error("Must have one or more filters.");
        }

        this.outputDimensions = inputDimensions.slice();
        for (let i = 0; i < 2; i++) {
            let dimension = this.filterDimensions[i];
            if (dimension % 2 == 0 || dimension < 1) {
                throw new Error("Filter dimensions must be odd and > 0.");
            }
            this.outputDimensions[i] -= dimension-1; // valid padding
        }
        this.outputDimensions[2] *= this.numFilters;  // each input 'image' is filtered once by each filter

        this.minValue = options.minValue ? options.minValue : -1;
        this.maxValue = options.minValue ? options.minValue :  1;

        this.filters = [];
        for (let i = 0; i < this.numFilters; i++) {
            let filter = new Matrix(this.filterSize, this.filterSize);
            filter.map(() => {
                return this.minValue + Math.random()*(this.maxValue - this.minValue);
            })
            this.filters.push(filter);
        }

        this.outputs = [];
        for (let i = 0; i < this.outputDimensions[2]; i++) {
            this.outputs.push(new Matrix(this.outputDimensions[0], this.outputDimensions[1]));
        }
        
    }

    /**
     * 
     * @param {Matrix[]} inputs 
     */
    processInputs(inputs) {

        let outputIndex = 0;
        for (let i = 0; i < inputs.length; i++) {  // loop through each input 'image'
            
            let input = inputs[i];
            for (let filter of this.filters) {

                let rowOffset = (filter.rows - 1)/2,
                    colOffset = (filter.cols - 1)/2;

                let outputMatrix = this.outputs[outputIndex];

                for (let row = 0; row < this.outputDimensions[0]; row++) {
                    for (let col = 0; col < this.outputDimensions[1]; col++) {

                        let vals = [];
                        for (let filterRow = 0; filterRow < this.filterDimensions[0]; filterRow++) {
                            for (let filterCol = 0; filterCol < this.filterDimensions[1]; filterCol++) {
                                
                                let actualRow = row + rowOffset + filterRow,
                                    actualCol = col + colOffset + filterCol;

                                let newVal = input.get(actualRow, actualCol) * filter.get(filterRow, filterCol);
                                outputMatrix.set(row, col, newVal);                  

                            }
                        }

                    }
                }

                outputIndex++;

            }
 
        }

    }

}

class PoolingLayer {

    constructor(inputDimensions, options) {

        if (!inputDimensions) {
            throw new Error("Input dimensions not specified.");
        }
        this.inputDimensions = inputDimensions.slice();

        this.filterDimensions = options.filterDimensions ? options.filterDimensions : [3, 3];
        if (this.filterDimensions.length !== 2) {
            throw new Error("Filter can only be 2D");
        }
        this.outputDimensions = inputDimensions.slice();
        for (let i = 0; i < 2; i++) {
            let dimension = this.filterDimensions[i];
            if (dimension < 1) {
                throw new Error("Filter dimensions must be > 0.");
            } else if (this.inputDimensions[i] % dimension != 0) {
                throw new Error("Filter does not evenly divide input.");
            }
            this.outputDimensions[i] /= dimension;
        }

        this.type = options.type ? options.type : LayerConstants.MAX_POOLING;
        if (this.type == LayerConstants.MAX_POOLING) {
            this.poolingFn = function(arraySegment) {
                // assumes arraySegment is a flattened array
                let max = -Infinity;
                for (let val of arraySegment) {
                    if (val > max) {
                        max = val;
                    }
                }
                return max;
            }
        } else if (this.type == LayerConstants.MIN_POOLING) {
            this.poolingFn = function(arraySegment) {
                let min = Infinity;
                for (let val of arraySegment) {
                    if (val < min) {
                        min = val;
                    }
                }
                return min;
            }
        } else if (this.type == LayerConstants.AVG_POOLING){
            this.poolingFn = function(arraySegment) {
                let sum = 0;
                for (let val of arraySegment) {
                    sum += val;
                }
                return sum / arraySegment.length;
            }
        } else {
            throw new Error("Invalid pooling type specified.");
        }

        this.outputs = []
        for (let i = 0; i < inputDimensions[2]; i++) {
            this.outputs.push(new Matrix(this.outputDimensions[0],
                                         this.outputDimensions[1]));
        }

    }

    processInputs(inputs) {

        for (let i = 0; i < inputs.length; i++) {  // loop through each input 'image'
            let input = inputs[i];
            for (let row = 0; row < this.outputDimensions[0]; row++) {
                for (let col = 0; col < this.outputDimensions[1]; col++) {

                    let vals = [];
                    for (let filterRow = 0; filterRow < this.filterDimensions[0]; filterRow++) {
                        for (let filterCol = 0; filterCol < this.filterDimensions[1]; filterCol++) {
                            vals.push(input.get(row*this.filterDimensions[0] + filterRow, 
                                                col*this.filterDimensions[1] + filterCol));
                        }
                    }
                    let newVal = this.poolingFn(vals);
                    this.outputs[i].set(row, col, newVal);

                }
            }
        }

    }

}

class LayerConstants {}

// pooling type constants
LayerConstants.MAX_POOLING = "max-pooling";
LayerConstants.MIN_POOLING = "min-pooling";
LayerConstants.AVG_POOLING = "avg-pooling";

// layer type constants
LayerConstants.FULLY_CONNECTED = "fully-connected";
LayerConstants.CONVOLUTIONAL = "convolutional";
LayerConstants.POOLING = "pooling";

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