class NeuralNetwork {

    /**
     * 
     * @param {number[]} neuronCountArray An array of numbers corresponding to the number of
     * neurons in each layer (first entry is input layer, last entry is output layer).
     */
     constructor(neuronCountArray, options) {

        this.neurons = [];
        for (let i = 0; i < neuronCountArray.length; i++) {
             this.neurons.push(Array.from(Array(neuronCountArray[i]), () => 0));
        }

        this.minBias = (options.minBias != undefined) ? options.minBias : -1;
        this.maxBias = (options.maxBias != undefined) ? options.maxBias : 1;

        // although biases are present for the inputs, this is superficial. it has no bearing
        // on any of the calculations, and was simply done to make indexing simpler.
        this.biases = this.neurons.map((val) => val.map((_val) => this.minBias + Math.random()*(this.maxBias - this.minBias)));

        this.minWeight = (options.minWeight != undefined) ? options.minWeight : -1;
        this.maxWeight = (options.maxWeight != undefined) ? options.maxWeight : 1;

        this.weights = [];
        for (let i = 0; i < this.neurons.length-1; i++) {
            let numCols = this.neurons[i].length,
                numRows = this.neurons[i+1].length;
            this.weights[i] = new Matrix(numRows, numCols);
            this.weights[i].map(() => this.minWeight + Math.random()*(this.maxWeight - this.minWeight));  // randomly initialise all weights
        }

        this.hiddenActivationType = (options.hiddenActivationType) ? options.hiddenActivationType : 
                                                                     NeuralNetwork.SIGMOID;
        if (this.hiddenActivationType == NeuralNetwork.SIGMOID) {
            this.hiddenActivationFn        = sigmoid;
            this.dHiddenActivationFn       = dSigmoid;
            this.hiddenGDFn                = oneToOneActivationGD;
        } else if (this.hiddenActivationType == NeuralNetwork.TANH) {
            this.hiddenActivationFn        = tanh;
            this.dHiddenActivationFn       = dTanh;
            this.hiddenGDFn                = oneToOneActivationGD;
        } else if (this.hiddenActivationType == NeuralNetwork.RELU) {
            this.hiddenActivationFn        = relu;
            this.dHiddenActivationFn       = dRelu;
            this.hiddenGDFn                = oneToOneActivationGD;
        } else {
            throw new Error("Invalid hidden activation function specified (note that softmax cannot be used for hidden layers).");
        }

        this.outputActivationType = (options.outputActivationType) ? options.outputActivationType : 
                                                                     NeuralNetwork.SIGMOID;
        if (this.outputActivationType == NeuralNetwork.SIGMOID) {
            this.outputActivationFn        = sigmoid;
            this.dOutputActivationFn       = dSigmoid;
            this.outputGDFn                = oneToOneActivationGD;
        } else if (this.outputActivationType == NeuralNetwork.TANH) {
            this.outputActivationFn        = tanh;
            this.dOutputActivationFn       = dTanh;
            this.outputGDFn                = oneToOneActivationGD;
        } else if (this.outputActivationType == NeuralNetwork.RELU){
            this.outputActivationFn        = relu;
            this.dOutputActivationFn       = dRelu;
            this.outputGDFn                = oneToOneActivationGD;
        } else if (this.outputActivationType == NeuralNetwork.SOFTMAX) {
            this.outputActivationFn        = softmax;
            this.outputGDFn                = crossEntropySoftmaxGD;
        } else {
            throw new Error("Invalid output activation function specified");
        }

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

    /**
     * Performs an iteration of the feed foward algorithm, from the layer whose index is specified by
     * "from" to the next layer (if none exists then the function simply returns - this is the base case).
     * @param {number[] | Matrix} values 
     * @param {number} from 
     */
    _feedForward(values, from) {

        if (from === 0) {  // "from" is 0
            if (values.length != this.neurons[0].length) {
                throw new Error("Number of inputs does not match number of input neurons");
            }
            this.neurons[0] = values.slice();  // copy array
        } else if (!(typeof from === "number")) {
            throw new Error("From must be passed, and must be a number.");
        }

        let weightsMatrix = this.weights[from];
        if (!weightsMatrix) {  // base case
            return;
        }

        let prevActivations;
        if (values instanceof Matrix) {

            if (weightsMatrix.cols != values.rows) {
                throw new Error(`Matrix ${values.rows + '*' + values.cols} 
                                 not compatible with matrix ${weightsMatrix.rows + '*' + weightsMatrix.cols}.`);
            } else {
                prevActivations = values;
            }

        } else if (values.length !== weightsMatrix.cols) {
            throw new Error(`Values only has ${values.length} elements but needs ${weightsMatrix.cols}.`);
        } else {
            prevActivations = Matrix.fromArray(values, Matrix.COLUMN);
        }
        
        let activationFn = (from === this.neurons.length-2) ?
                            this.outputActivationFn :
                            this.hiddenActivationFn;
        let newActivations = activationFn(Matrix.product(weightsMatrix, prevActivations));

        for (let i = 0; i < this.neurons[from+1].length; i++) {
            this.neurons[from+1][i] = newActivations.get(i, 0);
        }

        // recursive call
        this._feedForward(newActivations, from+1);

    }

    /**
     * Performs one step of the back-propagation algorithm.
     * @param {*} layerIndex The index of the layer we are currently propagating from
     */
    _backPropagation(layerIndex, _dErrorActivationMatrix, targets) {

        if (layerIndex < 1) {  // we have reached input layer
            return;
        }
        
        let neurons            = this.neurons[layerIndex],
            prevNeurons        = this.neurons[layerIndex-1];

        let biases = this.biases[layerIndex];

        let weights = this.weights[layerIndex-1],  // there is one more layer than weight matrices
            weightsT = Matrix.transpose(weights);

        let dErrorActivationMatrix;
        if (!_dErrorActivationMatrix) {  // we must be at the output layer, otherwise this would be passed to us from prev call
            
            if (!targets) { // we need target values to calculate error derivative
                throw new Error("Targets must be provided for output layer.");
            }
            dErrorActivationMatrix = this.dErrorFn(Matrix.fromArray(neurons, Matrix.COLUMN), 
                                                   Matrix.fromArray(targets, Matrix.COLUMN));
        } else {
            dErrorActivationMatrix = new Matrix(_dErrorActivationMatrix);
        }

        let {
            dErrorWeightMatrix,
            dErrorBiasMatrix,
            dErrorPrevActivationMatrix
        } = (_dErrorActivationMatrix) ? this.hiddenGDFn(neurons, prevNeurons, biases, weights, 
                                                        targets, dErrorActivationMatrix, this.dHiddenActivationFn) :
                                        this.outputGDFn(neurons, prevNeurons, biases, weights, 
                                                        targets, dErrorActivationMatrix, this.dOutputActivationFn)

        // subtract the calculated derivatives
        weights.sub(Matrix.mult(dErrorWeightMatrix, this.learningRate));
        this.biases[layerIndex] = biases.map((value, index) => {
            return value - dErrorBiasMatrix.get(index, 0)*this.learningRate;
        });

        // call recursively
        this._backPropagation(layerIndex-1, dErrorPrevActivationMatrix);

    }

    train(inputs, targets) {

        this._feedForward(inputs, 0);
        this._backPropagation(this.neurons.length-1, null, targets);
        return this.neurons[this.neurons.length-1];  // return guessed outputs

    }

    predict(inputs) {
        this._feedForward(inputs, 0);
        return this.neurons[this.neurons.length-1];
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