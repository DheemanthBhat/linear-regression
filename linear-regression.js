/**
 * Vectorized implementation of Linear-regression with Gradient-descent
 */
class LinearRegression {
  options = {
    learningRate: 0.1,
    iterations: 1000,
    batchSize: 10
  };

  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

    this.n = this.features.shape[0];
    this.c = this.features.shape[1];
    this.weights = tf.zeros([this.c, 1]);

    this.options = { ...this.options, ...options };
    this.mseHistory = [];
  }

  processFeatures(featuresArray) {
    let features = tf.tensor(featuresArray);
    features = this.standardize(features);

    const n = features.shape[0];
    const identity = tf.ones([n, 1]); // Identity vector
    // Concat features to identity vector.
    return identity.concat(features, 1);
  }

  normalEquation() {
    const xTranspose = this.features.transpose();

    const A = xTranspose.matMul(this.features);

    const AInverse = tf.tensor(math.inv(A.arraySync()));

    const theta = AInverse.matMul(xTranspose).matMul(this.labels);

    return theta;
  }

  // Mean Normalization
  standardize(features) {
    if (this.mean === undefined || this.standardDeviation === undefined) {
      const { mean, variance } = tf.moments(features, 0);
      this.mean = mean;
      this.standardDeviation = variance.pow(0.5);
    }

    return features.sub(this.mean).div(this.standardDeviation);
  }

  train() {
    console.log(`${this.n} records are used for training.`);
    const batchCount = parseInt(this.n / this.options.batchSize);
    console.log(`Batch Count: ${batchCount}`);
    for (let i = 0; i < this.options.iterations; i++) {
      // console.log(`Learning rate: ${this.options.learningRate}`);
      for (let j = 0; j < batchCount; j++) {
        const { features, labels } = this.getNextBatch(j);
        this.gradientDescent(features, labels);
      }

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  gradientDescent(features, labels) {
    // Total number of rows in features.
    const n = features.shape[0];

    /**
     * h = t0 + t1*x1 + t2*x2 + ... + tn*xn
     * h = X * Theta (Vector implementation)
     * Where:
     *  h - Hypothesis,
     *  x - features,
     *  X - feature matrix,
     *  Theta - parameters/weights matrix.
     */
    const h = features.matMul(this.weights);

    const difference = h.sub(labels);

    // Derivative of MSE w.r.t weights, i.e., J(Theta)
    const slopes = features
      .transpose()
      .matMul(difference)
      //.mul(2) // Optional (Usually this is omitted in most of the Gradient descent implementations).
      .div(n);

    /**
     * Multiply slopes with learning rate
     * and subtract results from weights.
     */
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  test(testFeaturesArray, testLabelsArray) {
    const testFeatures = this.processFeatures(testFeaturesArray);
    const testLabels = tf.tensor(testLabelsArray);

    const h = testFeatures.matMul(this.weights);
    const cod = this.rSquared(testLabels, h);
    // cod - coefficient of determination
    return cod;
  }

  rSquared(labels, hypothesis) {
    /**
     * Coefficient-of-determination or R-squared or (R^2)
     * R^2 = 1 - (SS_res / SS_tot)
     * Where:
     *  1. SS_res = Sum of Squares residual.
     *  2. SS_tot = Sum of Squares total.
     */
    const a = labels.sub(hypothesis);
    const SS_res = a.transpose().matMul(a).arraySync()[0][0];

    const b = labels.sub(labels.mean());
    const SS_tot = b.transpose().matMul(b).arraySync()[0][0];

    const R_squared = 1 - (SS_res / SS_tot);
    return R_squared;
  }

  // Record cost function
  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.n)
      .arraySync();

    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return;

    const currMSE = this.mseHistory[0];
    const prevMSE = this.mseHistory[1];
    // console.log(`CurrentMSE: ${currMSE}, prevMSE: ${prevMSE}`);
    if (currMSE > prevMSE) {
      // if MSE went up, divide learning rate by 2.
      this.options.learningRate /= 2;
    } else {
      // if MSE went down, increase learning rate by 5%.
      this.options.learningRate *= 1.05;
    }
  }

  getNextBatch(j) {
    const { batchSize } = this.options;
    const startIndex = j * batchSize;

    const features = this.features.slice([startIndex, 0], [batchSize, -1]);
    const labels = this.labels.slice([startIndex, 0], [batchSize, -1]);

    return { features, labels };
  }

  predict(observations) {
    const processedObservation = this.processFeatures(observations);
    return processedObservation.matMul(this.weights);
  }
}
