package ann

import (
	"log"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Pass represents layers that are pass-thru - layers that have external effects that do not affect the expression graph.
type Pass interface {
	PassThru()
}

// Metadata is not a real Layer. Its main aims is to extract metadata such as name or size from ConsOpts. This is useful in cases where the metadata needs to be composed as well.
// Note that the fields may end up being all empty.
type Metadata struct {
	name         string
	Size         int
	shape        tensor.Shape
	ActivationFn ActivationFunction

	//internal state
	upd uint // counts the number of times the data structure has been updated.
}

// Name returns the name. Conveniently, this makes *Metadata fulfil the Layer interface, so we may use it to extract the desired metadata.
// Unfortunately this also means that the name is not an exported field. A little inconsistency there.
func (m *Metadata) Name() string { return m.name }

// Shape will return the tensor.Shape of the metadata
func (m *Metadata) Shape() tensor.Shape { return m.shape }

// Describe will describe a metadata
func (m *Metadata) Describe() {}

// Model will return the gorgonia.Nodes associated with this metadata
func (m *Metadata) Model() G.Nodes { return nil }

// Fwd runs the equation forwards
func (m *Metadata) Fwd(x G.Input) G.Result { return G.Err(errors.New("Metadata is a dummy Layer")) }

// Type will return the hm.Type of the metadata
func (m *Metadata) Type() hm.Type { return nil }

// PassThru represents a passthru function
func (m *Metadata) PassThru() {}

// SetName allows for names to be set by a ConsOpt
func (m *Metadata) SetName(name string) error {
	if m.name != "" {
		return errors.Errorf("A name exists - %q ", m.name)
	}
	m.name = name
	m.upd++
	return nil
}

// SetSize allows for the metadata struct to be filled by a ConsOpt
func (m *Metadata) SetSize(size int) error {
	if m.Size != 0 {
		return errors.Errorf("A clashing size %d exists.", m.Size)
	}
	m.Size = size
	m.upd++
	return nil
}

// SetActivationFn allows the metadata to store activation function.
func (m *Metadata) SetActivationFn(act ActivationFunction) error {
	if m.ActivationFn != nil {
		return errors.New("A clashing activation function already exists")
	}
	m.ActivationFn = act
	m.upd++
	return nil
}

// ExtractMetadata extracts common metadata from a list of ConsOpts. It returns the metadata. Any unused ConsOpt is also returned.
// This allows users to selectively use the metadata and/or ConsOpt options
func ExtractMetadata(opts ...ConsOpt) (retVal Metadata, unused []ConsOpt, err error) {
	var l Layer = &retVal
	m := &retVal
	var ok bool
	upd := m.upd
	for _, opt := range opts {
		if l, err = opt(l); err != nil {
			return Metadata{}, unused, err
		}
		if m, ok = l.(*Metadata); !ok {
			return Metadata{}, unused, errors.Errorf("ConsOpt mutated metadata. Got %T instead", l)
		}
		if m.upd > upd {
			upd = m.upd
		} else {
			unused = append(unused, opt)
		}
	}

	return *m, unused, nil
}

// trace is a Layer used for debugging
type trace struct {
	name              string
	format, errFormat string
	logger            *log.Logger
}

// Trace creates a layer for debugging composed layers
//
// The format string adds four things: "%s %v (%p) %v" - name (of trace), x, x, x.Shape()
func Trace(name, format, errFormat string, logger *log.Logger) Term {
	const (
		def    = "\t%s %v (%p) %v"
		defErr = "\tERR %s %+v"
	)

	if format == "" {
		format = def
	}

	if errFormat == "" {
		errFormat = defErr
	}

	return &trace{
		name:      name,
		format:    format,
		errFormat: errFormat,
		logger:    logger,
	}
}

func (t *trace) Model() G.Nodes { return nil }
func (t *trace) Fwd(x G.Input) G.Result {
	err := G.CheckOne(x)
	var print func(string, ...interface{})

	print = log.Printf
	if t.logger != nil {
		print = t.logger.Printf
	}

	if err != nil {
		print(t.errFormat, t.name, err)
		return G.LiftResult(x, nil)
	}
	print(t.format, t.name, x, x, x.Node().Shape())
	return G.LiftResult(x, nil)
}
func (t *trace) Name() string        { return t.name }
func (t *trace) Type() hm.Type       { return nil }
func (t *trace) Shape() tensor.Shape { return nil }
func (t *trace) Describe()           {}
func (t *trace) PassThru()           {}
