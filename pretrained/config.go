package pretrained

// Config is an interface for pretrained model configuration.
// It has only one method `Load(string) error` to load configuration
// from local or remote file.
type Config interface {
	Load(modelNamOrPath string, params map[string]interface{}) error
}
