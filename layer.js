class FullyConnectedLayer {

    constructor(numNeurons, numInputs, options) {

        if (!numNeurons) {
            throw new Error("Number of neurons not specified.");
        } else if (!numInputs) {
            throw new Error("Number of inputs not specified.");
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

        this.paddingType = (options.paddingType) ? options.paddingType : LayerConstants.VALID_PADDING;

        this.outputDimensions = inputDimensions.slice();
        for (let i = 0; i < 2; i++) {
            let dimension = this.filterDimensions[i];
            if (dimension % 2 == 0 || dimension < 1) {
                throw new Error("Filter dimensions must be odd and > 0.");
            }
            if (this.paddingType === LayerConstants.VALID_PADDING) {
                this.outputDimensions[i] -= dimension-1;
            }
        }
        this.outputDimensions[2] *= this.numFilters;  // each input 'image' is filtered once by each filter

        this.minValue = options.minValue ? options.minValue : -1;
        this.maxValue = options.minValue ? options.minValue :  1;

        this.filters = [];
        for (let i = 0; i < this.numFilters; i++) {
            let filter = new Matrix(this.filterDimensions[0], this.filterDimensions[1]);
            filter.map(() => {
                return this.minValue + Math.random()*(this.maxValue - this.minValue);
            })
            this.filters.push(filter);
        }

        this.outputs = [];
        for (let i = 0; i < this.outputDimensions[2]; i++) {
            this.outputs.push(new Matrix(this.outputDimensions[0], this.outputDimensions[1]));
        }

        this.activationType = (options.activationType) ? options.activationType : LayerConstants.RELU;
        if (this.activationType === LayerConstants.RELU) {
            this.activationFn = relu;
        } else {
            throw new Error("relu or ur dumb bro");
        }

        this.activationPosition = (options.activationPosition) ? options.activationPosition : LayerConstants.BEFORE;
        
    }

    /**
     * 
     * @param {Matrix[]} inputs 
     */
    processInputs(inputs) {

        let outputIndex = 0;
        for (let filter of this.filters) {

            let rowOffset = (filter.rows - 1)/2,
                colOffset = (filter.cols - 1)/2;

            let n = filter.rows * filter.cols;
            
            for (let i = 0; i < inputs.length; i++) {

                let input = inputs[i];
                if (this.paddingType === LayerConstants.SAME_PADDING) {

                    // pad input with zeros
                    let newInput = new Matrix(input.rows + rowOffset*2, input.cols + colOffset*2);
                    for (let row = rowOffset; row < input.rows; row++) {
                        for (let col = colOffset; col < input.cols; col++) {
                            newInput.set(row, col, input.get(row - rowOffset, col - colOffset));
                        }
                    }
                    input = newInput;

                }
                let outputMatrix = this.outputs[outputIndex];

                for (let row = 0; row < this.outputDimensions[0]; row++) {
                    for (let col = 0; col < this.outputDimensions[1]; col++) {

                        let newVal = 0;
                        for (let filterRow = -rowOffset; filterRow     <= rowOffset; filterRow++) {
                            for (let filterCol = -colOffset; filterCol <= colOffset; filterCol++) {
                                
                                let actualRow = row + rowOffset + filterRow,
                                    actualCol = col + colOffset + filterCol;
                                
                                let inputVal = (this.activationPosition === LayerConstants.BEFORE) ? 
                                                this.activationFn(input.get(actualRow, actualCol)) :
                                                input.get(actualRow, actualCol);

                                newVal += inputVal * filter.get(filterRow + rowOffset, filterCol + colOffset);                 

                            }
                        }
                        newVal /= n;
                        if (this.activationPosition === LayerConstants.AFTER) {
                            newVal = this.activationFn(newVal);
                        }
                        outputMatrix.set(row, col, newVal); 

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

        this.filterDimensions = options.filterDimensions ? options.filterDimensions : [2, 2];
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

        this.outputs = [];
        this.contributionMatrices = [];  // keeps track of how 'responsible' each input value is for the output
        for (let i = 0; i < inputDimensions[2]; i++) {
            this.outputs.push(new Matrix(this.outputDimensions[0],
                                         this.outputDimensions[1]));
            this.contributionMatrices.push(new Matrix(this.inputDimensions[0],
                                                      this.inputDimensions[1]))
        }

    }

    processInputs(inputs) {

        for (let i = 0; i < inputs.length; i++) {  // loop through each input 'image'
            let input = inputs[i];
            for (let row = 0; row < this.outputDimensions[0]; row++) {
                for (let col = 0; col < this.outputDimensions[1]; col++) {

                    let newVal;
                    if (this.type === LayerConstants.MAX_POOLING || 
                        this.type === LayerConstants.MIN_POOLING) {
                        
                        let val;
                        let valRow, valCol;
                        for (let filterRow = 0; filterRow < this.filterDimensions[0]; filterRow++) {
                            for (let filterCol = 0; filterCol < this.filterDimensions[1]; filterCol++) {

                                let inputRow = row*this.filterDimensions[0] + filterRow,
                                    inputCol = col*this.filterDimensions[1] + filterCol;
                                
                                let inputVal = input.get(inputRow, inputCol);
                                if (!val) {

                                    val = inputVal;
                                    valRow = inputRow;
                                    valCol = inputCol;
                                    this.contributionMatrices[i].set(inputRow, inputCol, 1);

                                } else if ((this.type === LayerConstants.MIN_POOLING && inputVal < val) ||
                                           (this.type === LayerConstants.MAX_POOLING && inputVal > val)) {

                                    this.contributionMatrices[i].set(valRow, valCol, 0);
                                    this.contributionMatrices[i].set(inputRow, inputCol, 1);
                                    val = inputVal;
                                    valRow = inputRow;
                                    valCol = inputCol;

                                }
                                
                            }
                        }

                        newVal = val;

                    } else if (this.type === LayerConstants.AVG_POOLING) {

                        let sum = 0;
                        let n = this.filterDimensions[0] * this.filterDimensions[1];
                        for (let filterRow = 0; filterRow < this.filterDimensions[0]; filterRow++) {
                            for (let filterCol = 0; filterCol < this.filterDimensions[1]; filterCol++) {

                                let inputRow = row*this.filterDimensions[0] + filterRow,
                                    inputCol = col*this.filterDimensions[1] + filterCol;
                                
                                sum += input.get(inputRow, inputCol);
                                
                            }
                        }

                        for (let filterRow = 0; filterRow < this.filterDimensions[0]; filterRow++) {
                            for (let filterCol = 0; filterCol < this.filterDimensions[1]; filterCol++) {

                                let inputRow = row*this.filterDimensions[0] + filterRow,
                                    inputCol = col*this.filterDimensions[1] + filterCol;
                                
                                this.contributionMatrices[i].set(inputRow, inputCol, 1 / n);
                                
                            }
                        }

                        newVal = sum / n;

                    } else {
                        throw new Error("Invalid pooling type.");
                    }
                         
                     this.outputs[i].set(row, col, newVal);
 
                }
            }
        }

    }

}

class LayerConstants {}

// activation timing constants (for conv. layer)
LayerConstants.BEFORE = "before";
LayerConstants.AFTER  = "after";

// pooling type constants
LayerConstants.MAX_POOLING = "max-pooling";
LayerConstants.MIN_POOLING = "min-pooling";
LayerConstants.AVG_POOLING = "avg-pooling";

// padding type constants (for conv. layers)
LayerConstants.VALID_PADDING = "valid";
LayerConstants.SAME_PADDING = "same";

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