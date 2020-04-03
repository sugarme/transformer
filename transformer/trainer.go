package transformer

type LMTrainer interface {
	Train() (Model, error)
}
