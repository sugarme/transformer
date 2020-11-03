package transformer

import (
	"github.com/sugarme/gotch/nn"
	"github.com/sugarme/transformer/pretrained"
)

// LoadConfig loads pretrained model data from local or remote file.
//
// Parameters:
// - `model` pretrained Model (any model type that implements pretrained `Model` interface)
// - `modelNameOrPath` is a string of either
//		+ Model name or
// 		+ File name or path or
// 		+ URL to remote file
// If `modelNameOrPath` is resolved, function will cache data using `TransformerCache`
// environment if existing, otherwise it will be cached in `$HOME/.cache/transformers/` directory.
// If `modleNameOrPath` is valid URL, file will be downloaded and cached.
// Finally, model weights will be loaded to `varstore`.
func LoadModel(model pretrained.Model, modelNameOrPath string, config pretrained.Config, customParams map[string]interface{}, vs *nn.VarStore) error {
	return model.Load(modelNameOrPath, config, customParams, vs)
}
