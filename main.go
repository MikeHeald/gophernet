package gonet

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// neuralNet contains all of the information
// that defines a trained neural network.
type NeuralNet struct {
	config  NeuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense

    //added
    hiddenLayerInput       *mat.Dense
    hiddenLayerActivations *mat.Dense
    outputLayerInput       *mat.Dense
    networkError           *mat.Dense
    slopeOutputLayer       *mat.Dense
    slopeHiddenLayer       *mat.Dense
    dOutput                *mat.Dense
    dHiddenLayer           *mat.Dense
    wOutAdj                *mat.Dense
    wHiddenAdj             *mat.Dense
    errorAtHiddenLayer     *mat.Dense
    Output                 *mat.Dense
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type NeuralNetConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	NumEpochs     int
    LearningRate  float64
}

func main() {

	// Form the training matrices.
	inputs, labels := makeInputsAndLabels("data/train.csv")

	// Define our network architecture and learning parameters.
	config := NeuralNetConfig{
		InputNeurons:  4,
		OutputNeurons: 3,
		HiddenNeurons: 3,
		NumEpochs:     5000,
		LearningRate:  0.3,
	}

	// Train the neural network.
	network := NewNetwork(config)
    iRows, _ := inputs.Dims()
	if err := network.train(inputs, labels, iRows); err != nil {
		log.Fatal(err)
	}

	// Form the testing matrices.
	testInputs, testLabels := makeInputsAndLabels("data/test.csv")

	// Make the predictions using the trained model.
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy of our model.
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate the accuracy (subset accuracy).
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output the Accuracy value to standard out.
	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy)
}

func (nn *NeuralNet) GetOutput() []float64{
    return nn.Output.RawRowView(0)
}

// NewNetwork initializes a new neural network.
func NewNetwork(config NeuralNetConfig) *NeuralNet {
    nn := &NeuralNet{config: config,
        hiddenLayerInput: new(mat.Dense),
        hiddenLayerActivations: new(mat.Dense),
        outputLayerInput: new(mat.Dense),
        networkError: new(mat.Dense),
        slopeOutputLayer: new(mat.Dense),
        slopeHiddenLayer: new(mat.Dense),
        dOutput: new(mat.Dense),
        dHiddenLayer: new(mat.Dense),
        wOutAdj: new(mat.Dense),
        wHiddenAdj: new(mat.Dense),
        errorAtHiddenLayer: new(mat.Dense),
        Output: new(mat.Dense),

        wHidden: mat.NewDense(config.InputNeurons, config.HiddenNeurons, nil),
        bHidden: mat.NewDense(1, config.HiddenNeurons, nil),
        wOut: mat.NewDense(config.HiddenNeurons, config.OutputNeurons, nil),
        bOut: mat.NewDense(1, config.OutputNeurons, nil),
    }
    // Initialize biases/weights.
    randSource := rand.NewSource(time.Now().UnixNano())
    randGen := rand.New(randSource)

    wHiddenRaw := nn.wHidden.RawMatrix().Data
    bHiddenRaw := nn.bHidden.RawMatrix().Data
    wOutRaw := nn.wOut.RawMatrix().Data
    bOutRaw := nn.bOut.RawMatrix().Data

    for _, param := range [][]float64{
        wHiddenRaw,
        bHiddenRaw,
        wOutRaw,
        bOutRaw,
    } {
        for i := range param {
            param[i] = randGen.Float64()
        }
    }



    return nn
}

func (nn *NeuralNet) addBHidden(_, col int, v float64) float64 {
    return v + nn.bHidden.At(0, col)
}

func (nn *NeuralNet) applySigmoid(_, _ int, v float64) float64 {
    return sigmoid(v)
}

func (nn *NeuralNet) applySigmoidPrime(_, _ int, v float64) float64 {
    return sigmoidPrime(v)
}

func (nn *NeuralNet) addBOut(_, col int, v float64) float64 {
    return v + nn.bOut.At(0, col)
}


// train trains a neural network using backpropagation.
func (nn *NeuralNet) train(x, y *mat.Dense, steps int) error {
	// Use backpropagation to adjust the weights and biases.
    _, xCols := x.Dims()
    _, yCols := y.Dims()
    for i := 0; i < nn.config.NumEpochs; i++ {
        for step := 0; step < steps; step++{
            //get state
            //x = next line
            x1 := mat.NewDense(1, xCols, x.RawRowView(step))

            //feed forward
            nn.Feedforward(x1)

            //get y
            y1 := mat.NewDense(1, yCols, y.RawRowView(step))

            //backprop
            if err := nn.Backpropagate(x1, y1); err != nil {
                return err
            }
        }
    }

	return nil
}


func (nn *NeuralNet) Feedforward(x *mat.Dense) {
    nn.hiddenLayerInput.Mul(x, nn.wHidden)
    nn.hiddenLayerInput.Apply(nn.addBHidden, nn.hiddenLayerInput)

    nn.hiddenLayerActivations.Apply(nn.applySigmoid, nn.hiddenLayerInput)

    nn.outputLayerInput.Mul(nn.hiddenLayerActivations, nn.wOut)
    nn.outputLayerInput.Apply(nn.addBOut, nn.outputLayerInput)
    nn.Output.Apply(nn.applySigmoid, nn.outputLayerInput)
}

// backpropagate completes the backpropagation method.
func (nn *NeuralNet) Backpropagate(x, y *mat.Dense) error {
	// Complete the backpropagation.
	nn.networkError.Sub(y, nn.Output)

	nn.slopeOutputLayer.Apply(nn.applySigmoidPrime, nn.Output)
	nn.slopeHiddenLayer.Apply(nn.applySigmoidPrime, nn.hiddenLayerActivations)

    nn.dOutput.MulElem(nn.networkError, nn.slopeOutputLayer)
	nn.errorAtHiddenLayer.Mul(nn.dOutput, nn.wOut.T())

	nn.dHiddenLayer.MulElem(nn.errorAtHiddenLayer, nn.slopeHiddenLayer)

	// Adjust the parameters.
	nn.wOutAdj.Mul(nn.hiddenLayerActivations.T(), nn.dOutput)
	nn.wOutAdj.Scale(nn.config.LearningRate, nn.wOutAdj)
	nn.wOut.Add(nn.wOut, nn.wOutAdj)

	bOutAdj, err := sumAlongAxis(0, nn.dOutput)
	if err != nil {
		return err
	}
	bOutAdj.Scale(nn.config.LearningRate, bOutAdj)
	nn.bOut.Add(nn.bOut, bOutAdj)

	nn.wHiddenAdj.Mul(x.T(), nn.dHiddenLayer)
	nn.wHiddenAdj.Scale(nn.config.LearningRate, nn.wHiddenAdj)
	nn.wHidden.Add(nn.wHidden, nn.wHiddenAdj)

	bHiddenAdj, err := sumAlongAxis(0, nn.dHiddenLayer)
	if err != nil {
		return err
	}
	bHiddenAdj.Scale(nn.config.LearningRate, bHiddenAdj)
	nn.bHidden.Add(nn.bHidden, bHiddenAdj)

	return nil
}

// predict makes a prediction based on a trained
// neural network.
func (nn *NeuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Check to make sure that our neuralNet value
	// represents a trained model.
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Define the output of the neural network.
	nn.Output = new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	nn.Output.Apply(applySigmoid, outputLayerInput)

	return nn.Output, nil
}

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

// sumAlongAxis sums a matrix along a
// particular dimension, preserving the
// other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the dataset file.
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Create a new CSV reader reading from the opened file.
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Read in all of the CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold all the
	// float values that will eventually be
	// used to form matrices.
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// Will track the current index of matrix values.
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {

		// Skip the header row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns.
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to the labelsData if relevant.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}
