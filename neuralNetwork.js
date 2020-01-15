class NeuralNetwork {

    /**
     * 
     * @param {number[]} neuronCountArray An array of numbers corresponding to the number of
     * neurons in each layer (first entry is input layer, last entry is output layer).
     */
     constructor(options) {

        if (!options) {
            options = {};
        }

        this.layerBlueprints = [];
        this.layers = [];

        this.errorType = (options.errorType) ? options.errorType : 
                                               NeuralNetwork.SQUARED_ERROR;
        if (this.errorType == NeuralNetwork.SQUARED_ERROR) {
            this.errorFn = squaredError;
            this.dErrorFn = dSquaredError;
        } else if (this.errorType == NeuralNetwork.CROSS_ENTROPY) {
            this.errorFn = crossEntropy;
            this.dErrorFn = dCrossEntropy;
        } else {
            throw new Error("Invalid error function specified.");
        }

        this.learningRate = (options.learningRate) ? options.learningRate : 0.1;

    }

    addLayer(type, options) {
        this.layerBlueprints.push({
            type,
            options
        });
    }

    /**
     * Resets the layerBlueprints array to contain nothing (so you can start building a neural network from scratch)
     */
    resetLayers() {
        this.layerBlueprints = [];
    }

    compile() {

        if (!this.layerBlueprints) {
            throw new Error("No layers to compile!");
        }

        if (this.layerBlueprints[this.layerBlueprints.length-1].type !== LayerConstants.FULLY_CONNECTED) {
            throw new Error("Output layer must be a fully-connected layer.");
        }

        this.layers = [];

        for (let i = 0; i < this.layerBlueprints.length; i++) {

            let layerType = this.layerBlueprints[i].type;
            let layerOptions = this.layerBlueprints[i].options;
            let prevLayer = this.layerBlueprints[i-1];

            if (layerType === LayerConstants.FULLY_CONNECTED) {
                
                if (i === 0 && !layerOptions.numInputs) {  // first hidden layer
                    throw new Error("First hidden error must specify number of inputs!");
                }
                let numInputs;
                if (prevLayer) {
                    if (prevLayer.type === LayerConstants.FULLY_CONNECTED) {
                        numInputs = prevLayer.options.numNeurons;
                    } else {  // conv / pooling layer
                        // rows * cols * depth -> i.e. length of flattened matrices
                        numInputs = this.layers[i-1].outputDimensions.reduce((prevVal, currVal) => {
                            return prevVal * currVal;
                        }, 1);
                    }
                } else {
                    numInputs = layerOptions.numInputs
                }
                this.layers.push(new FullyConnectedLayer(layerOptions.numNeurons, numInputs, layerOptions));
            
            } else if (layerType === LayerConstants.CONVOLUTIONAL ||
                       layerType === LayerConstants.POOLING) {
                
                let inputDimensions;
                if (i === 0) {
                    if (!layerOptions.inputDimensions) {
                        throw new Error("First convolutional / pooling hidden layer must specify input dimensions.");
                    } 
                    inputDimensions = layerOptions.inputDimensions;
                } else {
                    if (prevLayer.type !== LayerConstants.CONVOLUTIONAL && 
                        prevLayer.type !== LayerConstants.POOLING) {
                        throw new Error("Convolutional / pooling layer must be preceded by a convolutional / pooling layer.");
                    }
                    inputDimensions = this.layers[i-1].outputDimensions;
                }

                let layer = layerType === LayerConstants.CONVOLUTIONAL ?
                            new ConvolutionalLayer(inputDimensions, layerOptions) :
                            new PoolingLayer(inputDimensions, layerOptions);
                this.layers.push(layer);

            } else {
                throw new Error("Invalid layer type (only fully-connected, pooling, and convolutional are supported atm.");
            }

        }

    }

    /**
     * Feeds the passed initial inputs forward through the neural network.
     * @param {Matrix[]} initInputs 
     */
    _feedForward(initInputs) {

        let inputs = initInputs;
        for (let i = 0; i < this.layers.length; i++) {

            let currLayer = this.layers[i];
            let prevLayer = this.layers[i-1];

            // if (i === 0 && prevActivations.rows != currLayer.numInputs) {
            //     throw new Error("Number of inputs does not match number of input neurons");
            // }

            if (currLayer instanceof ConvolutionalLayer ||
                currLayer instanceof PoolingLayer) {

                if (!prevLayer) {  // we are the first hidden layer

                    // check that input matches the expected dimensions
                    // (should be guaranteed for all layers past the first
                    // thanks to the compile() method)
                    if (inputs[0].rows != currLayer.inputDimensions[0] ||
                        inputs[0].cols != currLayer.inputDimensions[1] ||
                        inputs.length  != currLayer.inputDimensions[2]) {
                        throw new Error("Input dimensions do not match dimensions specified for the input layer.");
                    }         

                }

                currLayer.processInputs(inputs);
                inputs = currLayer.outputs;

            } else {  // fully-connected layer
                
                if (!(prevLayer instanceof FullyConnectedLayer)) {  // last layer was conv/pooling or does not exist
                    
                    let newInputs = inputs;
                    if (prevLayer) {  // prev layer does exists
                        // need to flatten inputs
                        newInputs = [];
                        for (let m of inputs) {
                            newInputs = newInputs.concat(m.to1DArray());
                        }
                    }

                    // then convert to column vector
                    inputs = Matrix.fromArray(newInputs, Matrix.COLUMN);

                }

                currLayer.neurons = currLayer.activationFn(Matrix.product(currLayer.weights, inputs));
                inputs = currLayer.neurons;

            }

        }

    }

    /**
     * Performs the back propagation algorithm
     * @param {number[] | Matrix} targets The target outputs
     */
    _backPropagation(targets) {

        let targetsMatrix = (targets instanceof Matrix) ? targets :
                                                          Matrix.fromArray(targets, Matrix.COLUMN);
        let outputLayer = this.layers[this.layers.length-1];

        let dErrorActivation = this.dErrorFn(outputLayer.neurons, targetsMatrix)
        for (let i = this.layers.length-1; i > 0; i--) {

            let currLayer = this.layers[i];
            let prevLayer = this.layers[i-1];

            if (currLayer instanceof FullyConnectedLayer) {

                let {
                    dErrorWeightMatrix,
                    dErrorBiasMatrix,
                    dErrorPrevActivationMatrix
                } = currLayer.gDFn(currLayer, prevLayer, targetsMatrix, dErrorActivation, currLayer.dActivationFn);
    
                // subtract the calculated derivatives
                currLayer.weights.sub(Matrix.mult(dErrorWeightMatrix, this.learningRate));
                currLayer.biases.sub(Matrix.mult(dErrorBiasMatrix, this.learningRate));
    
                dErrorActivation = dErrorPrevActivationMatrix;  // to be used for next layer

            } else if (currLayer instanceof PoolingLayer) {

                let dErrorPrevActivationMatrices = poolingGD(currLayer, prevLayer, dErrorActivation);
                dErrorActivation = dErrorPrevActivationMatrices;

            } else if (currLayer instanceof ConvolutionalLayer) {

                let {
                    dErrorPrevActivationMatrices,
                    dErrorFilterMatrices
                } = convolutionalGD(currLayer, prevLayer, dErrorActivation);

                for (let i = 0; i < dErrorFilterMatrices.length; i++) {
                    currLayer.filters[i].sub(Matrix.mult(dErrorFilterMatrices[i], this.learningRate));
                }

                dErrorActivation = dErrorPrevActivationMatrices;

            } else {

                throw new Error("invalid layer type");

            }

            

        }

    }

    train(inputs, targets) {

        if (!this.layers) {
            throw new Error("compile() must be called before training can take place.");
        }

        this._feedForward(inputs);
        this._backPropagation(targets);

        let outputLayer = this.layers[this.layers.length-1];
        return {
            guess: Matrix.to1DArray(outputLayer.neurons),
            error: this.errorFn(outputLayer.neurons, Matrix.fromArray(targets, Matrix.COLUMN))
        }

    }

    predict(inputs) {

        if (!this.layers) {
            throw new Error("compile() must be called before predictions can be made.");
        }

        this._feedForward(inputs);
        return Matrix.to1DArray(this.layers[this.layers.length-1].neurons);

    }

    toJSON() {

        return JSON.stringify(this);

    }

    static fromJSON(jsonString) {

        let obj = JSON.parse(jsonString);

        let weights = [];
        console.log(obj);
        for (let weightsObj of obj.weights) {
            let weightsMatrix = new Matrix(weightsObj.rows, weightsObj.cols);
            weightsMatrix.values = weightsObj.values;
            weights.push(weightsMatrix);
        }

        // can simply pass 'obj' for the options parameter as it will have all fields necessary
        let neuralNetwork = new NeuralNetwork([], obj);
        
        neuralNetwork.neurons = obj.neurons;
        neuralNetwork.biases  = obj.biases;
        neuralNetwork.weights = weights;

        return neuralNetwork;

    }

}

// activation constants
NeuralNetwork.SIGMOID = "sigmoid";
NeuralNetwork.TANH = "tanh";
NeuralNetwork.RELU = "relu";
NeuralNetwork.SOFTMAX = "softmax";

// error constants
NeuralNetwork.CROSS_ENTROPY = "cross-entropy";
NeuralNetwork.SQUARED_ERROR = "squared-error";