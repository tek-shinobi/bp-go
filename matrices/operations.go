package matrices

func numberlen(f float64) (res int) {
    for f >= 10 {
        f /= 10
        res++
    }
    return
}

// Negate negates its argument
func Negate(f float64) float64 {
    return -f
}

// OnePlus increments its argument
func OnePlus(f float64) float64 {
    return 1.0 + f
}

// OneMinus decrements its argument
func OneMinus(f float64) float64 {
    return 1.0 - f
}

// Invert inverts its argument
func Invert(f float64) float64 {
    return 1.0 / f
}

// Mult returns function that multiplies with given argument
func Mult(f float64) (func (float64) float64) {
    return func (g float64) float64 { return f * g; }
}

// Add returns function that adds given argument
func Add(f, g float64) (func (float64) float64) {
    return func (g float64) float64 { return f + g; }
}
