package nn

import "github.com/tek-shinobi/back-propagation-nn/matrices"

// TrainItem represents one item for training of neural network
type TrainItem struct {
	Values   matrices.Matrix
	Label    float64
	Distinct int
}

// InitTrainItem initializes new training item - values and label
func InitTrainItem(values []float64, label float64, distinct int) TrainItem {
	matrix := matrices.InitMatrixWithValues(len(values), values)
	return TrainItem{matrix, label, distinct}
}
