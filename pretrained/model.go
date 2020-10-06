package pretrained

import (
	"github.com/sugarme/gotch/nn"
)

// Model is an interface for pretrained model.
// It has only one method `Load(string) error` to load model
// from local or remote file.
type Model interface {
	Load(modelNamOrPath string, config interface{ Config }, params map[string]interface{}, vs nn.VarStore) error
}
