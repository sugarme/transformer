package roberta

import (
	// "fmt"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/bpe"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/processor"

	"github.com/sugarme/transformer/util"
)

// Tokenizer holds data for Roberta tokenizer.
type Tokenizer struct {
	*tokenizer.Tokenizer
}

// NewTokenizer creates a new Roberta tokenizer.
func NewTokenizer() *Tokenizer {
	tk := tokenizer.NewTokenizer(nil)
	return &Tokenizer{tk}
}

// Load loads Roberta tokenizer from pretrain vocab and merges files.
func (t *Tokenizer) Load(modelNameOrPath string, params map[string]interface{}) error {
	vocabFile, err := util.CachedPath("roberta-base", "vocab.json")
	if err != nil {
		return err
	}
	mergesFile, err := util.CachedPath("roberta-base", "merges.txt")
	if err != nil {
		return err
	}

	model, err := bpe.NewBpeFromFiles(vocabFile, mergesFile)
	if err != nil {
		return err
	}

	t.WithModel(model)

	bertNormalizer := normalizer.NewBertNormalizer(true, true, true, true)
	t.WithNormalizer(bertNormalizer)

	blPreTokenizer := pretokenizer.NewByteLevel()
	// blPreTokenizer.SetAddPrefixSpace(false)
	t.WithPreTokenizer(blPreTokenizer)

	var specialTokens []tokenizer.AddedToken
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<s>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<pad>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("</s>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<unk>", true))
	specialTokens = append(specialTokens, tokenizer.NewAddedToken("<mask>", true))
	t.AddSpecialTokens(specialTokens)

	postProcess := processor.DefaultRobertaProcessing()
	t.WithPostProcessor(postProcess)

	return nil
}
