package pretrained

type Tokenizer interface {
	Load(modelNamOrPath string, params map[string]interface{}) error
}
