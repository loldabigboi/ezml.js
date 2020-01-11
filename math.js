// activation functions / derivatives

/** SIGMOID */
function sigmoid(values) {
    return Matrix.map(values, (x) => {
        return 1 / (1 + Math.exp(-x));
    });
}

function dSigmoid(activations) {
    // y is the result of sigmoid(x)
    return Matrix.map(activations, (y) => {
        return y * (1 - y);
    });
}

/** TANH */
function tanh(values) {

    return Matrix.map(values, (x) => {
        let exp = Math.exp(2*x);
        if (exp == Infinity) {  // need this check otherwise we get NaN
            return 1;  // approx. equal to 1.
        }
        return (Math.exp(2*x) - 1) / (Math.exp(2*x) + 1);
    });

}

function dTanh(activations) {
    return Matrix.map((activations), (y) => {
        return 1 - Math.pow(y, 2);
    });
}

/** RELU */
function relu(values) {
    return Matrix.map(values, (x) => {
        return Math.max(0, x);
    })
}

function dRelu(activations) {
    return Matrix.map(activations, (y) => {
        if (y == 0) {
            return 0;
        } else {
            return 1;
        }
    });
}

/** SOFTMAX */
function softmax(outputs) {

    let max = -Infinity;
    for (let x of outputs.values) {
        if (x[0] > max) {
            max = x[0];
        }
    }

    let expSum = 0;
    for (let x of outputs.values) {
        expSum += Math.exp(x[0] - max);  // substract max value so we don't get NaN (stable softmax)
    }

    return Matrix.map(outputs, (x) => Math.exp(x - max) / expSum);

}


// dError functions

function dSquaredError(neuronsMatrix, targetsMatrix) {

    return Matrix.map(neuronsMatrix, (val, row, col) => {
        return val - targetsMatrix.get(row, col);
    }); 

}

function dCrossEntropy(neuronsMatrix, targetsMatrix) {

    return Matrix.map(targetsMatrix, (val, row, col) => {
        return -val / neuronsMatrix.get(row, 0);
    });

}


/** Gradient descent algorithms */

// Gradient descent for squared error with 1:1 activation function
// (i.e. each activation is only dependent upon the single corresponding logit)
// e.g. Sigmoid, Tanh or ReLU
function oneToOneActivationGD(neurons, prevNeurons, biases, weights, targets, dErrorActivationMatrix, dActivationFn) {

    let neuronsMatrix = Matrix.fromArray(neurons, Matrix.COLUMN);
    let prevNeuronsMatrix = Matrix.fromArray(prevNeurons, Matrix.COLUMN);

    let dActivationZMatrix = dActivationFn(neuronsMatrix);

    // calculate dErrorWeightMatrix (how each weight affects error)
    let dErrorWeightMatrix = Matrix.map(weights, (val, row, col) => {

        let dZWeight        = prevNeuronsMatrix.get(col, 0);
        let dActivationZ    = dActivationZMatrix.get(row, 0);
        let dErrorActivation = dErrorActivationMatrix.get(row, 0);
        return dZWeight * dActivationZ * dErrorActivation;  // chain rule
    
    });

    // calculate dErrorBiasMatrix (how each bias affects error)
    // neuronsMatrix will have same dimensions as the bias matrix, so no need
    // to create a bias matrix
    let dErrorBiasMatrix = Matrix.map(neuronsMatrix, (val, row, col) => {
        
        let dZBias = 1;
        let dActivationZ = dActivationZMatrix.get(row, 0);
        let dErrorActivation = dErrorActivationMatrix.get(row, 0);
        return dZBias * dActivationZ * dErrorActivation;
    
    })

    // calculate dErrorPrevActivation (how each previous activation affects error)
    let dErrorPrevActivationMatrix = Matrix.map(prevNeuronsMatrix, (val, row, col) => {
        
        let total = 0;
        for (let i = 0; i < neurons.length; i++) {  // i indexes all the weights for this prev. neuron
            let dZPrevActivation = weights.get(i, row);
            let dActivationZ = dActivationZMatrix.get(i, 0);
            let dErrorActivation = dErrorActivationMatrix.get(i, 0);
            total += dZPrevActivation * dActivationZ * dErrorActivation;
        }
        return total;

    });

    return {
        dErrorWeightMatrix,
        dErrorBiasMatrix,
        dErrorPrevActivationMatrix
    }

}

// Gradient descent iteration for cross entropy error with softmax activation
function crossEntropySoftmaxGD(neurons, prevNeurons, biases, weights, targets, dErrorActivationMatrix) {
        
    // variable naming: dXY means the partial derivative of X with respect to Y
    //                  dXYMatrix means a matrix of values corresponding to the above
    //                  dXYArray means an array of values corresponding to the above

    let neuronsMatrix = Matrix.fromArray(neurons, Matrix.COLUMN),
        prevNeuronsMatrix = Matrix.fromArray(prevNeurons, Matrix.COLUMN);

    let dActivationZMatrix = new Matrix(neurons.length, neurons.length);
    dActivationZMatrix.map((val, i, j) => {

        return i === j ? neurons[j] * (1 - neurons[j]) : 
                        -neurons[i] * neurons[j];

    });

    let dErrorZMatrix = Matrix.map(neuronsMatrix, (val, i, col) => {

        let sum = 0;
        for (let j = 0; j < neurons.length; j++) {
            let dErrorActivation = dErrorActivationMatrix.get(j, 0);
            let dActivationZ = dActivationZMatrix.get(i, j);
            sum += dErrorActivation * dActivationZ;
        }

        return sum;

    });    

    // calculate dErrorWeightMatrix (how each weight affects error)
    let dErrorWeightMatrix = Matrix.map(weights, (val, row, col) => {
        
        let dErrorZ = dErrorZMatrix.get(row, 0);
        let dZWeight = prevNeuronsMatrix.get(col, 0);
        return dErrorZ * dZWeight;  // chain rule
    
    });

    // calculate dErrorBiasMatrix (how each bias affects error)
    // turns out to be indentical to dErrorZMatrix (same formula as for dErrorWeight,
    // but we replace dZWeight with dZBias which === 1, thus we get dErrorZ * 1 => dErrorZ)
    let dErrorBiasMatrix = new Matrix(dErrorZMatrix);

    // calculate dErrorPrevActivation (how each previous activation affects error)
    let dErrorPrevActivationMatrix = Matrix.map(prevNeuronsMatrix, (val, row, col) => {
        
        let total = 0;
        for (let i = 0; i < neurons.length; i++) {  // i indexes all the weights for this prev. neuron
            let dZPrevActivation = weights.get(i, row);
            let dErrorZ = dErrorZMatrix.get(i, 0);
            total += dZPrevActivation * dErrorZ;
        }
        return total;

    });

    neuralNetwork

    return {
        dErrorWeightMatrix,
        dErrorBiasMatrix,
        dErrorPrevActivationMatrix
    }

}


/**
 * 
 * @param {Matrix} outputs 
 * @param {Matrix} targets 
 */
function squaredError(outputs, targets) {
    return Matrix.map(outputs, (val, row, col) => {
        return val - targets.get(row, col);
    }); 
}
