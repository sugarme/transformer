package bert

import (
	"github.com/sugarme/sermo/tokenizer"
	data "github.com/sugarme/sermo/util/data"
)

type BertTrainer struct {
	Dataset   data.MapDataset
	Tokenizer tokenizer.Tokenizer // pretrained tokenizer
	Model     BertModel           // pretrained model
}

func (bt *BertTrainer) Train() (BertModel, error) {
	return BertModel{}, nil
}

func maskTokens(inputs Tensor, tokenizer tokenizer.Tokenizer) []Tensor {

}
