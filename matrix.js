class Matrix {

    /**
     * If a Matrix object is passed as the "rows" parameter, a deep-copy of said
     * Matrix instance will be created. Otherwise, a rows*cols matrix will be created
     * with all values initialised to zero.
     * @param {number | Matrix} rows 
     * @param {number} cols 
     */
    constructor(rows, cols) {

        this.values = [];

        if (rows instanceof Matrix) {

            let other = rows;
            this.rows = other.rows;
            this.cols = other.cols;

            // copy values (we cannot slice whole array as the sub-arrays will be shallow-copied)
            this.values = other.values.map((val) => val.slice());

        } else {

            this.rows = rows;
            this.cols = cols;

            // initialise all values to zero
            for (let row = 0; row < rows; row++) {
                let rowArr = [];
                for (let col = 0; col < cols; col++) {
                    rowArr[col] = 0;
                }
                this.values[row] = rowArr;
            }

        }

    }

    /**
     * Sets the value at the passed indices to the passed value.
     * @param {number} row 
     * @param {number} col 
     * @param {number} value 
     */
    set(row, col, value) {
        this.values[row][col] = value;
    }

    /**
     * Gets the value at the specified indices.
     * @param {number} row 
     * @param {number} col 
     */
    get(row, col) {
        return this.values[row][col];
    }

    /**
     * Adds the passed argument from this matrix, performing pairwise
     * addition if a Matrix is passed, and otherwise simply adding
     * the passed argument (scalar) to all elements.
     * @param {number | Matrix} value 
     */
    add(value) {

        if (value instanceof Matrix) {

            let other = value;
            if (other.rows != this.rows || other.cols != this.cols) {
                throw new Error("Matrix dimensions must match.");
            }

            // add all elements pairwise
            for (let row = 0; row < this.rows; row++) {
                for (let col = 0; col < this.cols; col++) {
                    this.values[row][col] += other.values[row][col]
                }
            }

        } else {

            // add scalar to all elements
            for (let row = 0; row < this.rows; row++) {
                for (let col = 0; col < this.cols; col++) {
                    this.values[row][col] += value;
                }
            }

        }

    }

    /**
     * Subtracts the passed argument from this matrix, performing pairwise
     * subtraction if a Matrix is passed, and otherwise simply subtracting
     * the passed argument (scalar) from all elements.
     * @param {number | Matrix} value 
     */
    sub(value) {

        if (value instanceof Matrix) {

            let other = value;
            if (other.rows !== this.rows || other.cols !== this.cols) {
                throw new Error("Matrix dimensions must match.");
            }

            // sub all elements pairwise
            for (let row = 0; row < this.rows; row++) {
                for (let col = 0; col < this.cols; col++) {
                    this.values[row][col] -= other.values[row][col];
                }
            }

        } else {

            // sub scalar from all elements
            for (let row = 0; row < this.rows; row++) {
                for (let col = 0; col < this.cols; col++) {
                    this.values[row][col] -= value;
                }
            }

        }

    }

    /**
     * Multiplies all elements in this matrix by the passed scalar, or performs pair-wise
     * multiplication with the passed matrix.
     * @param {number | Matrix} value 
     */
    mult(value) {

        let valFn = function() {
            return value;
        }
        if (value instanceof Matrix) {

            if (value.rows != this.rows || value.cols != this.cols) {
                throw new Error("Matrix dimensions must match.");
            }

            valFn = function(row, col) {
                return value.get(row, col);
            }

        }

        // multiply all elements by a scalar
        for (let row = 0; row < this.rows; row++) {
            for (let col = 0; col < this.cols; col++) {
                this.values[row][col] *= valFn(row, col);
            }
        }

    }

    /**
     * Transposes this matrix in place.
     */
    transpose() {

        let newValues = [];
        for (let col = 0; col < this.cols; col++) {
            newValues[col] = [];
            for (let row = 0; row < this.rows; row++) {
                newValues[col][row] = this.values[row][col];
            }
        }
        
        // swap rows and columns
        this.rows += this.cols;
        this.cols = this.rows - this.cols;
        this.rows = this.rows - this.cols;

        this.values = newValues;

    }
    /**
     * Calls the callback once for each value in this matrix
     * @param {function} callback The callback function
     */
    forEach(callback) {
        for (let row = 0; row < this.values.length; row++) {
            for (let col = 0; col < this.values[0].length; col++) {
                callback(this.values[row][col], row, col);
            }
        }
    }

    /**
     * Replaces each value in this matrix with the result of calling the callback
     * on each old value (same idea as Array.prototype.map);
     * @param {function} callback The callback function
     */
    map(callback) {

        for (let row = 0; row < this.values.length; row++) {
            for (let col = 0; col < this.values[0].length; col++) {
                this.values[row][col] = callback(this.values[row][col], row, col);
            }
        }

    }

    /**
     * Prints the values in this matrix to console in a table format
     */
    print() {
        console.table(this.values);
    }

    /**
     * Static version of mult() (returns new Matrix instance)
     * @param {Matrix} m1 
     * @param {Matrix | number} value 
     */
    static mult(m1, m2) {

        let result = new Matrix(m1);
        result.mult(m2);
        return result;

    }

    /**
     * Returns the matrix product of m1 times m2 (not the hadamard product)
     * @param {Matrix} m1 
     * @param {Matrix} m2 
     */
    static product(m1, m2) {

        if (!(m1 instanceof Matrix) || !(m2 instanceof Matrix)) {
            throw new Error("Arguments must be of type Matrix.");
        } else if (m1.cols != m2.rows) {
            throw new Error("Matrix dimensions must match.");
        }

        // compute matrix product
        let result = new Matrix(m1.rows, m2.cols);
        for (let row = 0; row < m1.rows; row++) {
            for (let col2 = 0; col2 < m2.cols; col2++) {
                let dot = 0;
                for (let col1 = 0; col1 < m1.cols; col1++) {
                    // we can use col1 to index the row of m2 as #cols in m1 == #rows in m2
                    dot += m1.values[row][col1]*m2.values[col1][col2]
                }
                result.values[row][col2] = dot;
            }
        }

        return result;

    }

    /**
     * Returns a Matrix created from the passed 1D array
     * @param {number[]} array Array to create matrix from
     * @param {number} type Specifies type of matrix to create, either Matrix.ROW (vector) or Matrix.COLUMN (vector)
     */
    static fromArray(array, type) {

        let result;
        if (type === Matrix.COLUMN) {

            result = new Matrix(array.length, 1);
            for (let i = 0; i < array.length; i++) {
                result.values[i][0] = array[i];
            }

        } else {
            result = new Matrix(1, array.length);
            result.values = [array.slice()];
        }

        return result;

    }

    /**
     * Returns the result of transposing the passed matrix (passed matrix not changed)
     * @param {Matrix} m The matrix to transpose
     */
    static transpose(m) {

        let result = new Matrix(m);
        result.transpose();
        return result;

    }

    /**
     * Static version of add() (returns new Matrix instance)
     * @param {Matrix} m1
     * @param {number | Matrix} m2
     */
    static add(m1, m2) {

        let result = new Matrix(m1);
        result.add(m2);
        return result;

    }

    /**
     * Static version of sub() (returns new Matrix instance)
     * @param {Matrix} m1
     * @param {number | Matrix} m2
     */
    static sub(m1, m2) {

        let result = new Matrix(m1);
        result.sub(m2);
        return result;

    }

    /**
     * Static version of mult() (returns new Matrix instance)
     * @param {Matrix} m
     * @param {number} scalar
     */
    static scalarMult(m, scalar) {

        let result = new Matrix(m);
        result.mult(scalar);
        return result;

    }

    /**
     * Static version of map() (returns new Matrix instance)
     * @param {Matrix} m
     * @param {function} callback
     */
    static map(m, callback) {

        let newMatrix = new Matrix(m);
        newMatrix.map(callback);
        return newMatrix;

    }

    /**
     * Converts the passed matrix to a one-dimensional array.
     * @param {Matrix} m
     */
    static to1DArray(m) {

        return this.values.flat(Infinity);

    }

}
// constants used for specifying type of fromArray result (row or column vector)
Matrix.ROW = 0;
Matrix.COLUMN = 1;
