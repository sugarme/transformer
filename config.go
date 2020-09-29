package transformer

import (
	"fmt"

	"github.com/sugarme/transformer/util"
)

// PretrainedConfig is an interface for pretrained model configuration.
// It has only one method `Load(string) error` to load configuration
// from local or remote file.
type PretrainedConfig interface {
	Load(modelNamOrPath string, params map[string]interface{}) error
}

// LoadConfig loads configuration data from local or remote file.
//
// Parameters:
// - `config` PretrainedConfig (any model config that implement `PretrainedConfig` interface)
// - `modelNameOrPath` is a string of either
//		+ Model name or
// 		+ File name or path or
// 		+ URL to remote file
// If `modelNameOrPath` is resolved, function will cache data using `TransformerCache`
// environment if existing, otherwise it will be cached in `$HOME/.cache/transformer/` directory.
// If `modleNameOrPath` is valid URL, file will be downloaded and cached.
// Finally, configuration data will be loaded to `config` parameter.
func LoadConfig(config PretrainedConfig, modelNameOrPath string, customParams map[string]interface{}) error {
	// path, err := getFromCache(modelNameOrPath)
	// if err != nil {
	// return err
	// }

	err := config.Load(modelNameOrPath, customParams)
	if err != nil {
		return err
	}

	return nil
}

func getFromCache(modelNameOrPath string) (path string, err error) {
	localFile := "bert-base-uncased-config.json"
	path = fmt.Sprintf("%s/%s", util.DefaultCachePath, localFile)

	// TODO: implement it
	return path, nil
}
