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


// Error functions

function squaredError(neuronsMatrix, targetsMatrix) {

    let sum = 0;
    neuronsMatrix.forEach((val, row, col) => {
        sum += Math.pow(val - targetsMatrix.get(row, col), 2);
    }); 
    return sum;

}

function dSquaredError(neuronsMatrix, targetsMatrix) {

    return Matrix.map(neuronsMatrix, (val, row, col) => {
        return val - targetsMatrix.get(row, col);
    }); 

}

function crossEntropy(neuronsMatrix, targetsMatrix) {

    let sum = 0;
    neuronsMatrix.forEach((val, row, col) => {
        sum -= Math.log(val) * targetsMatrix.get(row, col);
    });
    return sum;

}

function dCrossEntropy(neuronsMatrix, targetsMatrix) {

    return Matrix.map(targetsMatrix, (val, row, col) => {
        return -val / neuronsMatrix.get(row, 0);
    });

}

/**
 * Gradient descent algorithm for convolutional layers
 * @param {PoolingLayer | ConvolutionalLayer} currLayer 
 * @param {PoolingLayer | ConvolutionalLayer} prevLayer 
 * @param {Matrix | Matrix[]} dErrorActivationMatrices A Nx1 matrix (passed from fully-connected layer) 
 * or array of 2D error matrices (passed from conv/pool layer)
 */
function convolutionalGD(currLayer, prevLayer, dErrorActivationMatrices) {

    // prevLayer must, according to the compile() method, be a pooling
    // or convolutional layer

    let outputRows = currLayer.outputDimensions[0],
        outputCols = currLayer.outputDimensions[1];

    let errorMatrices;

    if (!Array.isArray(dErrorActivationMatrices)) {
        // convert 1D errors to 3D representation
        for (let depth = 0; depth < currLayer.outputDimensions[2]; depth++) {

            let errorMatrix = new Matrix(currLayer.outputDimensions[0],
                                        currLayer.outputDimensions[1]);
            
            let iStart = depth*outputRows*outputCols;
            for (let i = iStart; i <iStart + outputRows*outputCols; i++) {
                let row = Math.floor(i / outputCols),
                    col = i % outputCols;
                errorMatrix.set(row, col, dErrorActivationMatrix.get(i, 0));
            }
            errorMatrices.push(errorMatrix);

        }
    } else {
        errorMatrices = dErrorActivationMatrices;
    }

    let dErrorPrevActivationMatrices = [];
    for (let i = 0; i < currLayer.inputDimensions[2]; i++) {
        dErrorPrevActivationMatrices.push(new Matrix(currLayer.inputDimensions[0], currLayer.inputDimensions[1]));
    }

    let dErrorFilterMatrices = [];

    let outputIndex = 0;
    for (let filter of currLayer.filters) {  // loop through all filters

        let rowOffset = (filter.rows - 1)/2,
            colOffset = (filter.cols - 1)/2;

        let dErrorFilterMatrix = new Matrix(filter.rows, filter.cols);
        let n = filter.rows * filter.cols;
        
        for (let i = 0; i < currLayer.inputDimensions[2]; i++) {  // loop through each input matrix
            
            let inputMatrix = prevLayer.outputs[i];
            let dErrorActivationMatrix = errorMatrices[outputOffset + i];
            let dErrorPrevActivationMatrix = dErrorPrevActivationMatrices[i];

            for (let row = 0; row < this.outputDimensions[0]; row++) {
                for (let col = 0; col < this.outputDimensions[1]; col++) {

                    let inputRow = row + rowOffset,
                        inputCol = col + colOffset;

                    for (let filterRow = -rowOffset; filterRow     <= rowOffset; filterRow++) {
                        for (let filterCol = -colOffset; filterCol <= colOffset; filterCol++) {
                            
                            let actualRow = inputRow + filterRow,
                                actualCol = inputCol + filterCol;

                            let dErrorActivation = dErrorActivationMatrix.get(row, col);
                            let dActivationFilter = inputMatrix.get(actualRow, actualCol) / n;
                            let dErrorFilter = dErrorFilterMatrix.get(rowOffset + filterRow, colOffset + filterCol) +  
                                               dErrorActivation * dActivationFilter;

                            let dActivationPrevActivation = filter.get(rowOffset + filterRow, colOffset + filterCol);
                            let dErrorPrevActivation = dErrorPrevActivationMatrix.get(inputRow, inputCol) + 
                                                       dErrorActivation * dActivationPrevActivation;

                             dErrorFilterMatrix.set(rowOffset + filterRow, colOffset + filterCol, dErrorFilter); 
                             dErrorPrevActivationMatrix.set(inputRow, inputCol, dErrorPrevActivation);    

                        }
                    }

                }
            }

            outputIndex++;

        }

        dErrorFilterMatrices.push(dErrorFilterMatrix);

    }

    return {
        dErrorFilterMatrices,
        dErrorPrevActivationMatrices
    }

}
/**
 * Gradient descent algorithm for pooling layers
 * @param {PoolingLayer | ConvolutionalLayer} currLayer 
 * @param {PoolingLayer | ConvolutionalLayer} prevLayer 
 * @param {Matrix | Matrix[]} dErrorActivationMatrix A Nx1 matrix or array of 2D error matrices (passed from conv/pool layer)
 */
function poolingGD(currLayer, prevLayer, dErrorActivationMatrices) {

    // prevLayer must, according to the compile() method, be a pooling
    // or convolutional layer

    let outputRows = currLayer.outputDimensions[0],
        outputCols = currLayer.outputDimensions[1];

    let errorMatrices;

    if (!Array.isArray(dErrorPrevActivationMatrices)) {
        // convert 1D errors to 3D representation
        for (let depth = 0; depth < currLayer.outputDimensions[2]; depth++) {

            let errorMatrix = new Matrix(currLayer.outputDimensions[0],
                                        currLayer.outputDimensions[1]);
            
            let iStart = depth*outputRows*outputCols;
            for (let i = iStart; i <iStart + outputRows*outputCols; i++) {
                let row = Math.floor(i / outputCols),
                    col = i % outputCols;
                errorMatrix.set(row, col, dErrorActivationMatrix.get(i, 0));
            }
            errorMatrices.push(errorMatrix);

        }
    } else {
        errorMatrices = dErrorActivationMatrices;
    }
    

    let filterWidth  = currLayer.filterDimensions[0],
        filterHeight = currLayer.filterDimensions[1];

    let dErrorPrevActivationMatrices = [];
    for (let i = 0; i < currLayer.inputDimension[2]; i++) {

        let dErrorPrevActivationMatrix = new Matrix(currLayer.contributionMatrices[i]);
        let errorMatrix = errorMatrices[i];

        for (let outputRow = 0; outputRow < outputRows; outputRow++) {
            for (let outputCol = 0; outputCol < outputCols; outputCol++) {

                let inputRow = outputRow*filterHeight,
                    inputCol = outputCol*filterWidth;

                for (let rowOffset = 0; rowOffset < filterHeight; rowOffset++) {
                    for (let colOffset = 0; colOffset < filterWidth; colOffset++) {
                        // set error for given input based upon its contribution to the corresponding output
                        let contribution = dErrorPrevActivationMatrix.get(inputRow + rowOffset, inputCol + colOffset);
                        let error = errorMatrix.get(outputRow, outputCol);
                        dErrorPrevActivationMatrix.set(inputRow + rowOffset, inputCol + colOffset, contribution*error);
                    }
                }

            }
        }
        
        dErrorPrevActivationMatrices.push(dErrorPrevActivationMatrix);

    }

    return dErrorPrevActivationMatrices;

}

/* Gradient descent algorithms for fully-connected layers */

// Gradient descent for squared error with 1:1 activation function
// (i.e. each activation is only dependent upon the single corresponding logit)
// e.g. Sigmoid, Tanh or ReLU
function oneToOneActivationGD(currLayer, prevLayer, targets, dErrorActivationMatrix, dActivationFn) {

    let prevNeuronsMatrix;
    if (prevLayer instanceof FullyConnectedLayer) {
        prevNeuronsMatrix = prevLayer.neurons;
    } else { // flatten all prev. outputs into a column vector
        prevNeurons = [];
        for (let input of prevLayer.outputs) {
            prevNeurons = prevNeurons.concat(input.to1DArray());
        }
        prevNeuronsMatrix = Matrix.fromArray(prevNeurons, Matrix.COLUMN);
    }

    let dActivationZMatrix = dActivationFn(currLayer.neurons);

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
        for (let i = 0; i < neuronsMatrix.rows; i++) {  // i indexes all the weights for this prev. neuron
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
function crossEntropySoftmaxGD(currLayer, prevLayer, targets, dErrorActivationMatrix) {
        
    // variable naming: dXY means the partial derivative of X with respect to Y
    //                  dXYMatrix means a matrix of values corresponding to the above
    //                  dXYArray means an array of values corresponding to the above

    let prevNeuronsMatrix;
    if (prevLayer instanceof FullyConnectedLayer) {
        prevNeuronsMatrix = prevLayer.neurons;
    } else { // flatten all prev. outputs into a column vector
        prevNeurons = [];
        for (let input of prevLayer.outputs) {
            prevNeurons = prevNeurons.concat(input.to1DArray());
        }
        prevNeuronsMatrix = Matrix.fromArray(prevNeurons, Matrix.COLUMN);
    }
    let neuronsMatrix = currLayer.neurons;

    let dActivationZMatrix = new Matrix(neuronsMatrix.rows, neuronsMatrix.rows);
    dActivationZMatrix.map((val, i, j) => {

        return i === j ? neuronsMatrix.get(j, 0) * (1 - neuronsMatrix.get(j, 0)) : 
                        -neuronsMatrix.get(i, 0) * neuronsMatrix.get(j, 0);

    });

    let dErrorZMatrix = Matrix.map(neuronsMatrix, (val, i, col) => {

        let sum = 0;
        for (let j = 0; j < neuronsMatrix.rows; j++) {
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
        for (let i = 0; i < neuronsMatrix.rows; i++) {  // i indexes all the weights for this prev. neuron
            let dZPrevActivation = weights.get(i, row);
            let dErrorZ = dErrorZMatrix.get(i, 0);
            total += dZPrevActivation * dErrorZ;
        }
        return total;

    });

    return {
        dErrorWeightMatrix,
        dErrorBiasMatrix,
        dErrorPrevActivationMatrix
    }

}
