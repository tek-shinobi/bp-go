package matrices

import (
    "fmt"
    "math"
    "math/rand"
    "errors"
    "strings"
    "encoding/json"
)

// Matrix represents two-dimensional field
type Matrix struct {
    cols int
    values []float64
}

// Rows returns number of rows in matrix
func (m Matrix) Rows() int {
    return len(m.values) / m.cols
}

// Cols returns number of columns in matrix
func (m Matrix) Cols() int {
    return m.cols
}

// InitMatrix initializes Matrix structure to have required number of rows and columns
func InitMatrix(rows, cols int) Matrix {
    m := Matrix{cols: cols}
    m.values = make([]float64, rows*cols)
    return m
}

// RandInitMatrix initializes Matrix structure and fills it with random numbers
func RandInitMatrix(rows, cols int) Matrix {
    m := InitMatrix(rows, cols)
    for i := range m.values {
        m.values[i] = rand.NormFloat64()
    }
    return m
}

// RandInitMatrixNormalized initializes Matrix structure and fills it with random numbers with respect to rows count
func RandInitMatrixNormalized(rows, cols int) Matrix {
    m := InitMatrix(rows, cols)
    for i := range m.values {
        m.values[i] = rand.NormFloat64() / math.Sqrt(float64(rows))
    }
    return m
}

// InitMatrixWithValues initializes Matrix with given dimensions and values
func InitMatrixWithValues(cols int, values []float64) Matrix {
    return Matrix{cols: cols, values: values}
}

// OneHotMatrix creates matrix that has one on given position and zeros everywhere else
func OneHotMatrix(rows, cols, setrow, setcol int) (Matrix, error) {
    m := InitMatrix(rows, cols)
    err := m.Set(setrow, setcol, 1.0)
    return m, err
}

// Copy creates copy of given matrix
func (m Matrix) Copy() Matrix {
    vals := make([]float64, len(m.values))
    copy(vals, m.values)
    return InitMatrixWithValues(m.cols, vals)
}

func (m Matrix) checkRowCol(row, col int) bool {
    return row < m.Rows() && col < m.Cols() && row >= 0 && col >= 0
}

func (m Matrix) at(row, col int) float64 {
    return m.values[row * m.Cols() + col]
}

func (m Matrix) set(row, col int, value float64) {
    m.values[row * m.Cols() + col] = value
}

// At returns item that is in matrix at given coordinates
func (m Matrix) At(row, col int) (float64, error) {
    if !m.checkRowCol(row, col) {
        return 0, errors.New("matrices: cannot get value outside of matrix")
    }
    return m.at(row, col), nil
}

// Set sets item in matrix to given value
func (m Matrix) Set(row, col int, value float64) error {
    if !m.checkRowCol(row, col) {
        return errors.New("matrices: cannot set value outside of matrix")
    }
    m.set(row, col, value)
    return nil
}

func (m Matrix) operate(n Matrix, operation func(float64, float64) float64) (Matrix, error) {
    var result Matrix
    if m.Rows() != n.Rows() || m.Cols() != n.Cols() {
        return result, errors.New("matrices: operating on two matrices with different dimensions")
    }
    result = InitMatrix(m.Rows(), m.Cols())
    for i := range m.values {
        result.values[i] = operation(m.values[i], n.values[i])
    }
    return result, nil
}

// Add adds two matrices
func (m Matrix) Add(n Matrix) (Matrix, error) {
    return m.operate(n, func (x, y float64) float64 { return x + y; })
}

// Sub subtracts two matrices
func (m Matrix) Sub(n Matrix) (Matrix, error) {
    return m.operate(n, func (x, y float64) float64 { return x - y; })
}

// Mult multiplies elements in matrices piecewise
func (m Matrix) Mult(n Matrix) (Matrix, error) {
    return m.operate(n, func (x, y float64) float64 { return x * y; })
}

// Apply applies function to each element of Matrix
func (m Matrix) Apply(operation func(float64) float64) Matrix {
    result := InitMatrix(m.Rows(), m.Cols())
    for i, val := range m.values {
        result.values[i] = operation(val)
    }
    return result
}

// Sum sumarizes whole matrix
func (m Matrix) Sum() float64 {
    sum := 0.0
    for _, val := range m.values {
        sum += val
    }
    return sum
}

// Dot multiplies two matrices
func (m Matrix) Dot(n Matrix) (Matrix, error) {
    var result Matrix
    if m.Cols() != n.Rows() {
        return result, errors.New("matrices: for matrix multiplication, first matrix cols == second matrix rows")
    }
    result = InitMatrix(m.Rows(), n.Cols())
    for i := 0; i < result.Rows(); i++ {
        for j := 0; j < result.Cols(); j++ {
            sum := 0.0
            for counter := 0; counter < m.Cols(); counter++ {
                sum += m.at(i, counter) * n.at(counter, j)
            }
            result.set(i, j, sum)
        }
    }
    return result, nil
}

// Transpose creates transposed matrix of original matrix
func (m Matrix) Transpose() Matrix {
    result := InitMatrix(m.Cols(), m.Rows())
    for i := 0; i < m.Rows(); i++ {
        for j := 0; j < m.Cols(); j++ {
            val := m.at(i, j)
            result.set(j, i, val)
        }
    }
    return result
}

// Max returns biggest value in matrix
func (m Matrix) Max() (float64, error) {
    index, err := m.MaxAt()
    return m.values[index], err
}

// MaxAt returns index where biggest value in matrix is
func (m Matrix) MaxAt() (int, error) {
    if m.Rows() == 0 || m.Cols() == 0 {
        return 0, errors.New("matrices: can't return max value in empty matrix")
    }
    maxval := m.values[0]
    maxvalIndex := 0
    for i, val := range m.values {
        if val > maxval {
            maxval = val
            maxvalIndex = i
        }
    }
    return maxvalIndex, nil
}

// Min returns smallest value in matrix
func (m Matrix) Min() (float64, error) {
    index, err := m.MinAt()
    return m.values[index], err
}

// MinAt returns index where smallest value in matrix is
func (m Matrix) MinAt() (int, error) {
    if m.Rows() == 0 || m.Cols() == 0 {
        return 0, errors.New("matrices: can't return min value in empty matrix")
    }
    minval := m.values[0]
    minvalIndex := 0
    for i, val := range m.values {
        if val < minval {
            minval = val
            minvalIndex = i
        }
    }
    return minvalIndex, nil
}

// Sigmoid returns Matrix where Sigmoid function was applied to each element
func (m Matrix) Sigmoid() Matrix {
    return m.Apply(Negate).Apply(math.Exp).Apply(OnePlus).Apply(Invert)
}

// SigmoidPrime returns Matrix where SigmoidPrime function was applied to each element
func (m Matrix) SigmoidPrime() Matrix {
    result, err := m.Sigmoid().Mult(m.Sigmoid().Apply(OneMinus))
    if err != nil {
        panic(err)
    }
    return result
}

func (m Matrix) String() (result string) {
    maxval, err := m.Max()
    if err != nil {
        panic(err)
    }
    flen := numberlen(maxval)
    floatfmt := fmt.Sprintf("%%%d.2f", flen + 6)
    rows := make([]string, m.Rows())

    for i := 0; i < m.Rows(); i++ {
        row := ""
        for j := 0; j < m.Cols(); j++ {
            val := m.at(i, j)
            row += fmt.Sprintf(floatfmt, val)
        }
        rows[i] = "| " + row + " |"
    }
    result = strings.Join(rows, "\n")
    return
}

// MarshalJSON implements Marshaler interface
func (m Matrix) MarshalJSON() ([]byte, error) {
    res := struct {
        Cols int
        Values []float64
    }{
        m.cols,
        m.values,
    }
    return json.Marshal(res)
}

// UnmarshalJSON implements Unmarshaler interface
func (m *Matrix) UnmarshalJSON(serialized []byte) error {
    var exportedMatrix struct {
        Cols int
        Values []float64
    }
    if err := json.Unmarshal(serialized, &exportedMatrix); err != nil {
        return err
    }
    m.cols = exportedMatrix.Cols
    m.values = exportedMatrix.Values
    return nil
}
