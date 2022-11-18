package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"

	"github.com/tek-shinobi/back-propagation-nn/matrices"
)

// NN represents neural network to be used with backpropagation
type NN struct {
	layers  []int
	weights []matrices.Matrix
	biases  []matrices.Matrix
}

// InitNN creates new neural network with given number of layers, neurons in each layer and initalizes them randomly
func InitNN(layers []int) NN {
	biases := make([]matrices.Matrix, len(layers)-1)
	weights := make([]matrices.Matrix, len(layers)-1)

	for i := range layers[1:] {
		biases[i] = matrices.RandInitMatrix(1, layers[i+1])
	}

	for i := range layers[1:] {
		weights[i] = matrices.RandInitMatrixNormalized(layers[i], layers[i+1])
	}

	return NN{layers, weights, biases}
}

// Copy creates copy if given network
func (network NN) Copy() NN {
	layers := make([]int, len(network.layers))
	copy(layers, network.layers)
	biases := make([]matrices.Matrix, len(network.biases))
	for i, bias := range network.biases {
		biases[i] = bias.Copy()
	}
	weights := make([]matrices.Matrix, len(network.weights))
	for i, weight := range network.weights {
		weights[i] = weight.Copy()
	}
	return NN{layers, biases, weights}
}

func (network NN) String() (result string) {
	result = "Neural network:\n"
	result += "layers:"
	for _, layer := range network.layers {
		result += fmt.Sprintf(" %d", layer)
	}
	for i, weights := range network.weights {
		result += fmt.Sprintf("\nweights layer %d to %d:\n%s", i+1, i, weights.String())
	}
	for i, biases := range network.biases {
		result += fmt.Sprintf("\nbiases layer %d:\n%s", i+1, biases.String())
	}

	return
}

// FeedForward returns output of given Network on given input
func (network NN) FeedForward(input matrices.Matrix) matrices.Matrix {
	lastOutput := input
	for i := range network.weights {
		weights := network.weights[i]
		biases := network.biases[i]
		multiplied, err := lastOutput.Dot(weights)
		if err != nil {
			panic(err)
		}
		added, err := multiplied.Add(biases)
		if err != nil {
			panic(err)
		}
		lastOutput = added.Sigmoid()
	}
	return lastOutput
}

// Evaluate returns ratio of correctly clasified inputs
func (network NN) Evaluate(inputs []TrainItem) float64 {
	correct := 0
	for _, input := range inputs {
		output := network.FeedForward(input.Values)
		max, err := output.MaxAt()
		if err != nil {
			panic(err)
		}
		if float64(max) == input.Label {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

// Cost returns total cost of input training items for cross-entropy
func (network NN) Cost(inputs []TrainItem) float64 {
	cost := 0.0
	for _, input := range inputs {
		output := network.FeedForward(input.Values)
		y, err := matrices.OneHotMatrix(1, input.Distinct, 0, int(input.Label))
		if err != nil {
			panic(err)
		}
		first, err := y.Apply(matrices.Negate).Mult(output.Apply(math.Log2))
		if err != nil {
			panic(err)
		}
		second, err := y.Apply(matrices.OneMinus).Mult(output.Apply(matrices.OneMinus).Apply(math.Log2))
		if err != nil {
			panic(err)
		}
		together, err := first.Sub(second)
		if err != nil {
			panic(err)
		}
		cost += together.Sum()
	}
	return cost / float64(len(inputs))
}

// Train trains Network on given input with given settings
func (network NN) Train(inputs []TrainItem, epochs, miniBatchSize int, eta, etaFraction, lmbda float64, testData []TrainItem, printCost bool) {
	oldEta := eta
	inputCount := len(inputs)
	i := 0
	doingBestOfN := false
	if epochs < 0 {
		doingBestOfN = true
		epochs = -epochs
	}
	bestCost := network.Cost(testData)
	bestNetwork := network.Copy()
	bestBefore := 0
	for {
		if !doingBestOfN && i >= epochs {
			break
		} else if doingBestOfN && bestBefore >= epochs {
			if etaFraction > 0 && eta*etaFraction > oldEta {
				bestBefore = 0
				eta /= 2.0
			} else {
				network = bestNetwork
				break
			}
		}
		shuffled := make([]TrainItem, inputCount)
		perm := rand.Perm(inputCount)
		for i, v := range perm {
			shuffled[v] = inputs[i]
		}

		batchesCount := int(float64(inputCount)/float64(miniBatchSize) + 0.5)
		batches := make([][]TrainItem, batchesCount)
		for i := 0; i < batchesCount; i++ {
			if i+miniBatchSize >= inputCount {
				batches[i] = shuffled[i*miniBatchSize:]
			} else {
				batches[i] = shuffled[i*miniBatchSize : i*miniBatchSize+miniBatchSize]
			}
		}

		for _, batch := range batches {
			network.updateMiniBatch(batch, eta, lmbda, len(inputs))
		}

		cost := network.Cost(testData)
		if doingBestOfN {
			if cost < bestCost {
				bestCost = cost
				bestNetwork = network.Copy()
				bestBefore = 0
			} else {
				bestBefore++
			}
		}

		if len(testData) > 0 {
			fmt.Printf("Epoch %d: %f\n", i, network.Evaluate(testData))
			if printCost {
				fmt.Printf("Cost: %f\n", cost)
			}
		} else {
			fmt.Printf("Epoch %d finished.\n", i)
		}
		i++
	}
}

func (network NN) updateMiniBatch(batch []TrainItem, eta, lmbda float64, n int) {
	var err error
	cxw := make([]matrices.Matrix, len(network.weights))
	cxb := make([]matrices.Matrix, len(network.biases))
	for i, m := range network.weights {
		cxw[i] = matrices.InitMatrix(m.Rows(), m.Cols())
	}
	for i, m := range network.biases {
		cxb[i] = matrices.InitMatrix(m.Rows(), m.Cols())
	}

	for _, item := range batch {
		nablaW, nablaB := network.backprop(item)
		for i, nabla := range nablaW {
			cxw[i], err = cxw[i].Add(nabla)
			if err != nil {
				panic(err)
			}
		}
		for i, nabla := range nablaB {
			cxb[i], err = cxb[i].Add(nabla)
			if err != nil {
				panic(err)
			}
		}
	}
	multByConst := matrices.Mult(eta / float64(len(batch)))
	for i, w := range cxw {
		regularization := matrices.Mult(1 - eta*lmbda/float64(n))
		reduced := w.Apply(multByConst)
		network.weights[i], err = network.weights[i].Apply(regularization).Sub(reduced)
		if err != nil {
			panic(err)
		}
	}
	for i, b := range cxb {
		reduced := b.Apply(multByConst)
		network.biases[i], err = network.biases[i].Sub(reduced)
		if err != nil {
			panic(err)
		}
	}
}

func (network NN) backprop(item TrainItem) ([]matrices.Matrix, []matrices.Matrix) {
	nablaW := make([]matrices.Matrix, len(network.weights))
	nablaB := make([]matrices.Matrix, len(network.biases))
	for i, m := range network.weights {
		nablaW[i] = matrices.InitMatrix(m.Rows(), m.Cols())
	}
	for i, m := range network.biases {
		nablaB[i] = matrices.InitMatrix(m.Rows(), m.Cols())
	}

	activation := item.Values
	activations := make([]matrices.Matrix, len(network.weights)+1)
	activations[0] = activation
	zs := make([]matrices.Matrix, len(network.weights))

	for i := range network.weights {
		weights := network.weights[i]
		biases := network.biases[i]
		multiplied, err := activation.Dot(weights)
		if err != nil {
			panic(err)
		}
		z, err := multiplied.Add(biases)
		if err != nil {
			panic(err)
		}
		zs[i] = z
		activation = z.Sigmoid()
		activations[i+1] = activation
	}

	y, err := matrices.OneHotMatrix(1, item.Distinct, 0, int(item.Label))
	if err != nil {
		panic(err)
	}

	// old code with MSE
	// costDerivative, err := activations[len(activations) - 1].Sub(y)
	// if err != nil {
	//     panic(err)
	// }
	// delta, err := costDerivative.Mult(zs[len(zs) - 1].SigmoidPrime())
	// if err != nil {
	//     panic(err)
	// }

	// new code with cross-entropy
	delta, err := activations[len(activations)-1].Sub(y)
	if err != nil {
		panic(err)
	}
	nablaB[len(nablaB)-1] = delta
	nablaW[len(nablaW)-1], err = activations[len(activations)-2].Transpose().Dot(delta)
	if err != nil {
		panic(err)
	}

	for l := 2; l < len(network.layers); l++ {
		z := zs[len(zs)-l]
		sp := z.SigmoidPrime()
		dotted, err := delta.Dot(network.weights[len(network.weights)-l+1].Transpose())
		if err != nil {
			panic(err)
		}
		delta, err = dotted.Mult(sp)
		if err != nil {
			panic(err)
		}
		nablaB[len(nablaB)-l] = delta
		nablaW[len(nablaW)-l], err = activations[len(activations)-l-1].Transpose().Dot(delta)
		if err != nil {
			panic(err)
		}
	}

	return nablaW, nablaB
}

// MarshalJSON implements Marshaler interface
func (network NN) MarshalJSON() ([]byte, error) {
	exportedNetwork := struct {
		Layers  []int
		Weights []matrices.Matrix
		Biases  []matrices.Matrix
	}{
		network.layers,
		network.weights,
		network.biases,
	}
	return json.Marshal(exportedNetwork)
}

// UnmarshalJSON implements Unmarshaler interface
func (network *NN) UnmarshalJSON(serialized []byte) error {
	var exportedNetwork struct {
		Layers  []int
		Weights []matrices.Matrix
		Biases  []matrices.Matrix
	}
	if err := json.Unmarshal(serialized, &exportedNetwork); err != nil {
		return err
	}
	network.layers = exportedNetwork.Layers
	network.weights = exportedNetwork.Weights
	network.biases = exportedNetwork.Biases
	return nil
}

// Save exports network to file as JSON
func (network NN) Save(path string) error {
	res, err := json.Marshal(network)
	if err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(res)
	return err
}

// LoadNetwork loads network from JSON file
func LoadNetwork(path string) (NN, error) {
	var network NN
	dat, err := ioutil.ReadFile(path)
	if err != nil {
		return network, err
	}

	err = json.Unmarshal(dat, &network)

	return network, err
}
