class NeuralNetwork {

    /**
     * 
     * @param {number[]} neuronCountArray An array of numbers corresponding to the number of
     * neurons in each layer (first entry is input layer, last entry is output layer).
     */
     constructor(options) {

        this.layerBlueprints = [];
        this.layers = [];

        this.errorType = (options.errorType) ? options.errorType : 
                                               NeuralNetwork.SQUARED_ERROR;
        if (this.errorType == NeuralNetwork.SQUARED_ERROR) {
            this.dErrorFn = dSquaredError;
        } else if (this.errorType == NeuralNetwork.CROSS_ENTROPY) {
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

        this.layers = [];

        for (let i = 0; i < this.layerBlueprints.length; i++) {

            let layerType = this.layerBlueprints[i].type;
            let layerOptions = this.layerBlueprints[i].options;
            let prevLayer = this.layerBlueprints[i-1];

            if (i === 0 && !layerOptions.numInputs) {  // first hidden layer
                throw new Error("First hidden error must specify number of inputs!");
            }

            let numInputs = (prevLayer) ? prevLayer.options.numNeurons : layerOptions.numInputs;

            if (layerType === LayerConstants.FULLY_CONNECTED) {
                this.layers.push(new FullyConnectedLayer(layerOptions.numNeurons, numInputs, layerOptions ));
            } else {
                throw new Error("Only supported layer type atm is fully-connected layers.");
            }

        }

    }

    /**
     * Feeds the passed inputs forwards through the neural network.
     * @param {number[] | Matrix} inputs 
     */
    _feedForward(inputs) {

        let prevActivations = (inputs instanceof Matrix) ? new Matrix(inputs) : 
                                                           Matrix.fromArray(inputs, Matrix.COLUMN);
        for (let i = 0; i < this.layers.length-1; i++) {

            let currLayer = this.layers[i];

            if (i === 0 && prevActivations.rows != currLayer.numInputs) {
                throw new Error("Number of inputs does not match number of input neurons");
            }

            currLayer.neurons = Matrix.product(currLayer.weights, prevActivations);
            prevActivations = currLayer.neurons;

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

        let dErrorActivationMatrix = this.dErrorFn(outputLayer.neurons, targetsMatrix)
        for (let i = this.layers.length-1; i > 0; i++) {

            let currLayer = this.layers[i];
            let nextLayer = this.layers[i-1];

            let {
                dErrorWeightMatrix,
                dErrorBiasMatrix,
                dErrorPrevActivationMatrix
            } = currLayer.gDFn(currLayer.neurons, nextLayer.neurons, currLayer.biases, currLayer.weights, 
                               targetsMatrix, dErrorActivationMatrix, currLayer.dActivationFn);

            // subtract the calculated derivatives
            currLayer.weights.sub(Matrix.mult(dErrorWeightMatrix, this.learningRate));
            currLayer.biases.sub(Matrix.mult(dErrorBiasMatrix, this.learningRate));

            dErrorActivationMatrix = dErrorPrevActivationMatrix;  // to be used for next layer

        }

    }

    train(inputs, targets) {

        if (!this.layers) {
            throw new Error("compile() must be called before training can take place.");
        }

        this._feedForward(inputs);
        this._backPropagation(targets);
        return Matrix.to1DArray(this.layers[this.layers.length-1].neurons);  // return guessed outputs

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