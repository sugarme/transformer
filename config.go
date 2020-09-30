package transformer

import (
	"github.com/sugarme/transformer/pretrained"
)

// LoadConfig loads pretrained configuration data from local or remote file.
//
// Parameters:
// - `config` pretrained.Config (any model config that implements pretrained `Config` interface)
// - `modelNameOrPath` is a string of either
//		+ Model name or
// 		+ File name or path or
// 		+ URL to remote file
// If `modelNameOrPath` is resolved, function will cache data using `TransformerCache`
// environment if existing, otherwise it will be cached in `$HOME/.cache/transformers/` directory.
// If `modleNameOrPath` is valid URL, file will be downloaded and cached.
// Finally, configuration data will be loaded to `config` parameter.
func LoadConfig(config pretrained.Config, modelNameOrPath string, customParams map[string]interface{}) error {
	return config.Load(modelNameOrPath, customParams)
}
